//! 连续批处理调度器（纯逻辑，无 GPU 依赖，可完全单测）。
//!
//! 策略：decode 优先 + prefill chunk 插入 + OOM 时抢占最年轻序列。

use ft_core::{FtError, ReqState, SeqId};
use ft_memory::kv::SeqEntry;
use ft_memory::KvPool;
use std::collections::VecDeque;
use std::time::Instant;

use crate::step::Step;

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// running batch 上限（对应 --max-running-requests）
    pub max_running: usize,
    /// prefill chunk 大小（对应 --max-prefill-length）
    pub chunk_tokens: usize,
    /// 每序列 KV token 下限（低于此值触发抢占）
    pub kv_reserve_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { max_running: 4, chunk_tokens: 8192, kv_reserve_tokens: 8192 }
    }
}

#[derive(Debug)]
pub struct RunningReq {
    pub state: ReqState,
    /// prefill 已完成的 token 数
    pub prefilled: usize,
    pub admitted_at: Instant,
}

#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub total_admitted: u64,
    pub total_finished: u64,
    pub total_preempted: u64,
    pub total_decode_tokens: u64,
}

pub struct Scheduler<K: KvPool> {
    cfg: SchedulerConfig,
    kv: K,
    queue: VecDeque<ReqState>,
    running: Vec<RunningReq>,
    stats: SchedulerStats,
    next_seq: u64,
}

impl<K: KvPool> Scheduler<K> {
    pub fn new(cfg: SchedulerConfig, kv: K) -> Self {
        Self { cfg, kv, queue: VecDeque::new(), running: Vec::new(), stats: SchedulerStats::default(), next_seq: 1 }
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
    pub fn running_len(&self) -> usize {
        self.running.len()
    }
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.running.is_empty()
    }

    /// 入队新请求，返回分配的 SeqId。
    pub fn submit(&mut self, mut req: ReqState) -> SeqId {
        req.seq = SeqId(self.next_seq);
        self.next_seq += 1;
        let sid = req.seq;
        self.queue.push_back(req);
        sid
    }

    /// 产生下一步动作。decode 优先；有空位则从 queue admit 并 prefill。
    pub fn next_step(&mut self) -> Result<Step, FtError> {
        // 1) 有 running 且都完成 prefill → decode
        if let Some(r) = self.running.iter_mut().find(|r| r.prefilled == r.state.prompt_tokens.len()) {
            let seqs = vec![r.state.seq];
            // 简化：单序列 decode（真实实现是整 batch 一步）
            let _ = r;
            self.stats.total_decode_tokens += seqs.len() as u64;
            return Ok(Step::Decode { seqs });
        }
        // 2) running 中有未完成 prefill → 继续 chunk
        if let Some(r) = self.running.iter().find(|r| r.prefilled < r.state.prompt_tokens.len()) {
            let (seq, start) = (r.state.seq, r.prefilled);
            let len = (r.state.prompt_tokens.len() - r.prefilled).min(self.cfg.chunk_tokens);
            let _ = self.running.iter_mut().find(|r| r.state.seq == seq).unwrap().prefilled += len;
            return Ok(Step::Prefill { seq, start, len });
        }
        // 3) 空 → admit
        if !self.queue.is_empty() && self.running.len() < self.cfg.max_running {
            let req = self.queue.pop_front().unwrap();
            let n = req.prompt_tokens.len();
            let entry = self.kv.alloc(req.seq, n)?;
            self.kv.stats(); // 触发统计
            let seq = req.seq;
            let len = n.min(self.cfg.chunk_tokens);
            self.running.push(RunningReq { state: req, prefilled: len, admitted_at: Instant::now() });
            let _ = entry;
            self.stats.total_admitted += 1;
            return Ok(Step::Prefill { seq, start: 0, len });
        }
        Ok(Step::Idle)
    }

    /// decode 产出 token 后回调：判断是否完成。
    pub fn on_decode_token(&mut self, seq: SeqId, token: u32) -> Option<ReqState> {
        let idx = self.running.iter().position(|r| r.state.seq == seq)?;
        let r = &mut self.running[idx];
        r.state.generated.push(token);
        if r.state.generated.len() >= r.state.max_new_tokens {
            let mut done = self.running.remove(idx);
            self.kv.free(SeqEntry { seq: done.state.seq, pages: vec![], num_tokens: 0 });
            done.state.kv_handle = None;
            self.stats.total_finished += 1;
            Some(done.state)
        } else {
            None
        }
    }

    /// OOM 时抢占 running 中最年轻的序列，塞回队列头部。
    pub fn preempt_youngest(&mut self) -> Option<Step> {
        let youngest = self
            .running
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.admitted_at)
            .map(|(i, _)| i)?;
        let r = self.running.remove(youngest);
        self.kv.free(SeqEntry { seq: r.state.seq, pages: vec![], num_tokens: 0 });
        let mut back = r.state;
        back.kv_handle = None;
        self.stats.total_preempted += 1;
        self.queue.push_front(back);
        Some(Step::Preempt { seq: SeqId(0) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ft_memory::SimpleKvPool;

    fn req(prompt: usize, max_new: usize) -> ReqState {
        ReqState {
            seq: SeqId(0),
            prompt_tokens: vec![0; prompt],
            kv_handle: None,
            generated: vec![],
            max_new_tokens: max_new,
        }
    }

    #[test]
    fn admit_prefill_then_decode_then_finish() {
        let mut s = Scheduler::new(SchedulerConfig::default(), SimpleKvPool::new(64));
        let sid = s.submit(req(10, 2));
        // admit + first prefill chunk
        match s.next_step().unwrap() {
            Step::Prefill { seq, start, len } => {
                assert_eq!((seq, start, len), (sid, 0, 10));
            }
            other => panic!("expected prefill, got {other:?}"),
        }
        // prefill 完成 → decode
        assert!(matches!(s.next_step().unwrap(), Step::Decode { .. }));
        // 第 1 个 token
        assert!(s.on_decode_token(sid, 7).is_none());
        // 第 2 个 token → 达到 max_new_tokens=2，请求完成返回
        let done = s.on_decode_token(sid, 8).unwrap();
        assert_eq!(done.generated, vec![7, 8]);
        assert!(s.is_idle());
        assert_eq!(s.stats().total_finished, 1);
    }

    #[test]
    fn chunked_prefill_splits_long_prompt() {
        let mut s = Scheduler::new(
            SchedulerConfig { chunk_tokens: 4, ..Default::default() },
            SimpleKvPool::new(64),
        );
        let sid = s.submit(req(10, 1));
        match s.next_step().unwrap() {
            Step::Prefill { seq, start, len } => assert_eq!((seq, start, len), (sid, 0, 4)),
            other => panic!("{other:?}"),
        }
        match s.next_step().unwrap() {
            Step::Prefill { seq, start, len } => assert_eq!((seq, start, len), (sid, 4, 4)),
            other => panic!("{other:?}"),
        }
        match s.next_step().unwrap() {
            Step::Prefill { seq, start, len } => assert_eq!((seq, start, len), (sid, 8, 2)),
            other => panic!("{other:?}"),
        }
        assert!(matches!(s.next_step().unwrap(), Step::Decode { .. }));
    }

    #[test]
    fn max_running_limits_admission() {
        let mut s = Scheduler::new(
            SchedulerConfig { max_running: 1, ..Default::default() },
            SimpleKvPool::new(64),
        );
        s.submit(req(1, 100));
        s.submit(req(1, 100));
        s.next_step().unwrap(); // admit #1（含首 chunk prefill）
        // #1 已完成 prefill → 下一步是 decode
        match s.next_step().unwrap() {
            Step::Decode { seqs } => assert_eq!(seqs.len(), 1),
            other => panic!("expected decode, got {other:?}"),
        }
        assert_eq!(s.queue_len(), 1);
        assert_eq!(s.running_len(), 1);
    }

    #[test]
    fn preempt_requeues() {
        let mut s = Scheduler::new(SchedulerConfig::default(), SimpleKvPool::new(64));
        s.submit(req(1, 100));
        s.next_step().unwrap();
        assert!(s.preempt_youngest().is_some());
        assert_eq!(s.queue_len(), 1);
        assert_eq!(s.running_len(), 0);
        assert_eq!(s.stats().total_preempted, 1);
    }
}
