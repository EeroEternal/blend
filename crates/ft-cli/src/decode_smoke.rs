//! 调度器 + SIMD MoE 内核的端到端集成冒烟。
//!
//! 走 ft-engine 的连续批处理循环：admit → chunked prefill → decode。
//! 每个 decode 步执行 `layers` 次真实 MoE 前向（模拟一层一层走完）。
use ft_core::{ReqState, SeqId};
use ft_engine::{Scheduler, SchedulerConfig, Step};
use ft_memory::SimpleKvPool;
use std::time::Instant;

pub fn run(steps: usize, layers: usize, full: bool, threads: usize) {
    let (hidden, inter, experts, k) = if full {
        (4096, 2048, 256, 6)
    } else {
        (256, 128, 16, 2)
    };
    let tokens = 1usize; // decode 关键路径：bs=1
    let threads = if threads == 0 { ft_moe::physical_cores() } else { threads };

    eprintln!(
        "decode-smoke: steps={steps} layers={layers} H{hidden} I{inter} E{experts} k{k} threads={threads}"
    );

    // 随机权重（simd 路径走 bf16）
    let mut state = 0xC0FFEE_u64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / u32::MAX as f32 - 0.5
    };
    let scale = 0.02f32 / (hidden as f32).sqrt();
    let w13_f: Vec<f32> = (0..experts * 2 * inter * hidden).map(|_| rnd() * scale).collect();
    let w2_f: Vec<f32> = (0..experts * hidden * inter).map(|_| rnd() * scale).collect();
    let f32_to_bf16 = |f: f32| -> u16 {
        let u = f.to_bits();
        let lsb = (u >> 16) & 1;
        ((u + 0x7fff + lsb) >> 16) as u16
    };
    let w13: Vec<u16> = w13_f.iter().map(|&v| f32_to_bf16(v)).collect();
    let w2: Vec<u16> = w2_f.iter().map(|&v| f32_to_bf16(v)).collect();
    let mut h: Vec<f32> = (0..tokens * hidden).map(|_| rnd()).collect();
    let topk: Vec<i32> = (0..tokens * k).map(|i| (i % experts) as i32).collect();
    let rw: Vec<f32> = vec![1.0 / k as f32; tokens * k];

    let mut sched = Scheduler::new(SchedulerConfig::default(), SimpleKvPool::new(256));
    let sid = sched.submit(ReqState {
        seq: SeqId(0),
        prompt_tokens: vec![1; 32],
        kv_handle: None,
        generated: vec![],
        max_new_tokens: steps,
    });

    let t0 = Instant::now();
    let mut ttft: Option<std::time::Duration> = None;
    let mut decode_tokens = 0usize;
    loop {
        match sched.next_step() {
            Ok(Step::Prefill { .. }) => { /* 本冒烟不跑真实 prefill 计算 */ }
            Ok(Step::Decode { seqs }) => {
                #[cfg(feature = "cpu-simd")]
                {
                    for _ in 0..layers {
                        ft_kernel::moe_bf16(
                            &mut h, &w13, &w2, &topk, &rw,
                            tokens, hidden, inter, experts, k, threads,
                        )
                        .expect("moe_bf16");
                    }
                }
                #[cfg(not(feature = "cpu-simd"))]
                {
                    let _ = (&w13, &w2, &h, layers, threads);
                }
                if ttft.is_none() {
                    ttft = Some(t0.elapsed());
                }
                decode_tokens += 1;
                if sched.on_decode_token(seqs[0], 7).is_some() {
                    break;
                }
                let _ = seqs;
            }
            Ok(Step::Idle) | Ok(Step::Preempt { .. }) | Ok(Step::Resize { .. }) => break,
            Err(e) => {
                eprintln!("scheduler error: {e}");
                break;
            }
        }
        let _ = sid;
    }
    let total = t0.elapsed();
    let decode_ms = if decode_tokens > 0 {
        (total.as_secs_f64() - ttft.unwrap_or(total).as_secs_f64()) * 1000.0
            / decode_tokens.max(1) as f64
    } else {
        0.0
    };
    println!(
        "decode-smoke: TTFT={:.1} ms, decode={:.1} ms/tok ({:.1} tok/s), tokens={decode_tokens}",
        ttft.unwrap_or(total).as_secs_f64() * 1000.0,
        decode_ms,
        if decode_ms > 0.0 { 1000.0 / decode_ms } else { 0.0 }
    );
}
