//! CPU MoE 吞吐基线。有效字节口径与 FreeToken benchbw 一致：
//! 每 token 每 expert 读 w13(2*I*H) + w2(H*I) 的 fp32 字节。
pub fn run(kernel: &str, tokens: usize, hidden: usize, inter: usize, experts: usize, k: usize, iters: usize) {
    use ft_moe::{CpuMoeExecutor, NaiveF32Executor};
    if kernel != "naive" && !cfg!(feature = "cpu-simd") {
        eprintln!("错误: kernel=simd 需要 --features cpu-simd 构建");
        std::process::exit(2);
    }
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

    // simd 路径需要 bf16 权重
    let f32_to_bf16 = |f: f32| -> u16 {
        let u = f.to_bits();
        let lsb = (u >> 16) & 1;
        (((u + 0x7fff + lsb) >> 16) as u16)
    };
    let w13_bf16: Vec<u16> = w13.iter().map(|&v| f32_to_bf16(v)).collect();
    let w2_bf16: Vec<u16> = w2.iter().map(|&v| f32_to_bf16(v)).collect();
    let topk_i32: Vec<i32> = topk.iter().map(|&v| v as i32).collect();

    let threads = ft_moe::physical_cores();
    if kernel == "simd" {
        #[cfg(feature = "cpu-simd")]
        println!("isa: {}", ft_kernel::cpu_isa_name());
        #[cfg(not(feature = "cpu-simd"))]
        unreachable!();
        println!("threads: {threads} (physical cores)");
    }

    // 预热
    match kernel {
        "naive" => NaiveF32Executor
            .forward(&mut h, tokens, hidden, &w13, &w2, inter, experts, &topk, &weights, k)
            .expect("warmup"),
        #[cfg(feature = "cpu-simd")]
        "simd" => ft_kernel::moe_bf16(
            &mut h, &w13_bf16, &w2_bf16, &topk_i32, &weights,
            tokens, hidden, inter, experts, k, threads,
        )
        .expect("warmup"),
        _ => unreachable!(),
    }

    // 按实际存储宽度计：naive=f32(4B), simd=bf16(2B)
    let elem_bytes = if kernel == "simd" { 2.0 } else { 4.0 };
    let bytes_per_step =
        (tokens * k) as f64 * ((2 * inter * hidden + hidden * inter) as f64) * elem_bytes;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        match kernel {
            "naive" => NaiveF32Executor
                .forward(&mut h, tokens, hidden, &w13, &w2, inter, experts, &topk, &weights, k)
                .expect("bench"),
            #[cfg(feature = "cpu-simd")]
            "simd" => ft_kernel::moe_bf16(
                &mut h, &w13_bf16, &w2_bf16, &topk_i32, &weights,
                tokens, hidden, inter, experts, k, threads,
            )
            .expect("bench"),
            _ => unreachable!(),
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let gbps = bytes_per_step * iters as f64 / dt / 1e9;
    println!(
        "{kernel}-moe-bench: {tokens}tok H{hidden} I{inter} E{experts} k{k}: {:.1} ms/step, effective {:.1} GB/s",
        dt * 1000.0 / iters as f64,
        gbps
    );
    std::hint::black_box(&h);
}
