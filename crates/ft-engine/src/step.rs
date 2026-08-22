//! 引擎单步动作的显式状态机（架构文档 §4.3）。

use ft_core::SeqId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// prefill 一个 chunk（chunked prefill，上限 max_extend_tokens）
    Prefill { seq: SeqId, start: usize, len: usize },
    /// decode 一步（整个 running batch）
    Decode { seqs: Vec<SeqId> },
    /// 抢占：把某序列换出（KV 页回收）
    Preempt { seq: SeqId },
    /// 弹性内存调整（对应 FreeToken elastic resize）
    Resize { moe_cache_slots: Option<usize>, kv_pages: Option<u32> },
    /// 空转
    Idle,
}

impl Step {
    pub fn is_idle(&self) -> bool {
        matches!(self, Step::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_detection() {
        assert!(Step::Idle.is_idle());
        assert!(!Step::Decode { seqs: vec![SeqId(1)] }.is_idle());
    }
}
