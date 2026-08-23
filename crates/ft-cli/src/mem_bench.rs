//! STREAM-like 多线程内存读带宽。标定 q* 画像的 cpu_gbps 输入。
//!
//! 方法：N 线程各自顺序求和独立 chunk（避免伪共享），总字节/时间。
pub fn run(gib: usize, iters: usize, threads: usize) {
    let threads = if threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
    } else {
        threads
    };
    let total = gib * (1 << 30);
    // 每 chunk 大小为 cache line 对齐的整数
    let per_thread = total / threads;
    let buf: Vec<u64> = vec![0x5555_5555_5555_5555u64; total / 8];

    // 预热一遍（触发页分配/缺页）
    sum_all(&buf, threads);

    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let sink = sum_all(&buf, threads);
        let dt = t0.elapsed().as_secs_f64();
        let gbps = total as f64 / dt / 1e9;
        best = best.min(gbps);
        std::hint::black_box(sink);
    }
    println!(
        "mem-bench: read {gib} GiB x {threads}t: best {:.1} GB/s",
        best
    );
}

fn sum_all(buf: &[u64], threads: usize) -> u64 {
    let per_thread = buf.len() / threads;
    std::thread::scope(|s| {
        let handles: Vec<_> = buf
            .chunks(per_thread)
            .map(|chunk| {
                s.spawn(move || {
                    let mut acc = 0u64;
                    for &v in chunk {
                        acc = acc.wrapping_add(v);
                    }
                    acc
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).fold(0u64, |a, b| a.wrapping_add(b))
    })
}
