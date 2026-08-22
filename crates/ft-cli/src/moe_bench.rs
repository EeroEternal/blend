//! CPU MoE 吞吐基线。有效字节口径与 FreeToken benchbw 一致：
//! 每 token 每 expert 读 w13(2*I*H) + w2(H*I) 的 fp32 字节。
pub fn run(tokens: usize, hidden: usize, inter: usize, experts: usize, k: usize, iters: usize) {
    use ft_moe::{CpuMoeExecutor, NaiveF32Executor};
    let mut state = 0x243F6A8885A308D3u64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / u32::MAX as f32 - 0.5
    };

    let scale = 0.02f32 / (hidden as f32).sqrt();
    let w13: Vec<f32> = (0..experts * 2 * inter * hidden).map(|_| rnd() * scale).collect();
    let w2: Vec<f32> = (0..experts * hidden * inter).map(|_| rnd() * scale).collect();
    let mut h: Vec<f32> = (0..tokens * hidden).map(|_| rnd()).collect();
    let topk: Vec<u32> = (0..tokens * k).map(|i| (i % experts) as u32).collect();
    let weights: Vec<f32> = (0..tokens * k).map(|_| 1.0 / k as f32).collect();

    let ex = NaiveF32Executor;
    // 预热
    ex.forward(&mut h, tokens, hidden, &w13, &w2, inter, experts, &topk, &weights, k)
        .expect("warmup");

    let bytes_per_step =
        (tokens * k) as f64 * ((2 * inter * hidden + hidden * inter) as f64) * 4.0;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        ex.forward(&mut h, tokens, hidden, &w13, &w2, inter, experts, &topk, &weights, k)
            .expect("bench");
    }
    let dt = t0.elapsed().as_secs_f64();
    let gbps = bytes_per_step * iters as f64 / dt / 1e9;
    println!(
        "naive-f32 moe-bench: {tokens}tok H{hidden} I{inter} E{experts} k{k}: {:.1} ms/step, effective {:.1} GB/s",
        dt * 1000.0 / iters as f64,
        gbps
    );
    std::hint::black_box(&h);
}
