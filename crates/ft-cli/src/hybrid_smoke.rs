//! Hybrid 路径冒烟：LRU 专家缓存 + q\* 拆分 + 真实 SIMD 内核。
//!
//! 模拟 FreeToken decode 的每层决策：
//!   命中 → 跳过（GPU 算，本冒烟计 0 成本）
//!   miss → q\* 拆成 Fetch（按 PCIe 带宽计时）+ CpuCompute（真实 moe_bf16）
//!
//! 这给出「全 CPU 6.3 tok/s」到「hybrid ~30 tok/s」之间的桥接估算。
use ft_moe::{
    BandwidthProfile, ExpertKey, LruExpertCache, MissAction, MoePlan, QStarPolicy,
};
use std::time::{Duration, Instant};

pub fn run(steps: usize, layers: usize, cache_slots: usize, threads: usize) {
    let (hidden, inter, experts, k) = (4096usize, 2048usize, 256usize, 6usize);
    let tokens = 1usize;
    let threads = if threads == 0 { ft_moe::physical_cores() } else { threads };
    let profile = BandwidthProfile::pro6000_epyc9355();
    let q = QStarPolicy::calibrate(&profile);
    let bytes_per_expert = ((2 * inter * hidden + hidden * inter) * 2) as f64; // bf16

    eprintln!(
        "hybrid-smoke: steps={steps} layers={layers} cache={cache_slots} slots, \
         fetch_frac={:.1}%, threads={threads}",
        q.fetch_fraction() * 100.0
    );

    let mut state = 0xA5A5_u64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let f32_to_bf16 = |f: f32| -> u16 {
        let u = f.to_bits();
        let lsb = (u >> 16) & 1;
        ((u + 0x7fff + lsb) >> 16) as u16
    };
    let scale = 0.02f32 / (hidden as f32).sqrt();
    let w13: Vec<u16> = (0..experts * 2 * inter * hidden)
        .map(|_| f32_to_bf16(((rnd() >> 40) as f32 / u32::MAX as f32 - 0.5) * scale))
        .collect();
    let w2: Vec<u16> = (0..experts * hidden * inter)
        .map(|_| f32_to_bf16(((rnd() >> 40) as f32 / u32::MAX as f32 - 0.5) * scale))
        .collect();
    let mut h: Vec<f32> = (0..tokens * hidden)
        .map(|_| (rnd() >> 40) as f32 / u32::MAX as f32 - 0.5)
        .collect();
    let rw: Vec<f32> = vec![1.0 / k as f32; tokens * k];

    let mut cache = LruExpertCache::new(cache_slots);
    // 每层上一 token 的路由，用于模拟时间局部性（论文图：8/12 hits from t-1）
    let mut prev: Vec<Vec<u32>> = vec![vec![0u32; k]; layers];
    let mut hits = 0u64;
    let mut fetches = 0u64;
    let mut cpu_misses = 0u64;
    let mut pcie_ns = 0u64;
    let mut cpu_ns = 0u64;

    let wall = Instant::now();
    for step in 0..steps {
        for layer in 0..layers {
            // 时间局部性：约 2/3 复用上一 token 的专家，其余替换（对齐 8/12 hits）
            let mut routed = prev[layer].clone();
            if step == 0 {
                for i in 0..k {
                    routed[i] = ((rnd() as usize).wrapping_add(layer * 17 + i * 31) % experts) as u32;
                }
            } else {
                let n_replace = (k + 2) / 3; // ≈ 2 of 6
                for i in 0..n_replace {
                    routed[i] = ((rnd() as usize).wrapping_add(layer * 13 + i * 7) % experts) as u32;
                }
            }
            prev[layer] = routed.clone();

            let is_cached = |e: u32| cache.contains(ExpertKey { layer: layer as u32, expert: e });
            let plan = MoePlan::build(&routed, is_cached, &q);
            let (h_n, f_n, c_n) = plan.stats();
            hits += h_n as u64;
            fetches += f_n as u64;
            cpu_misses += c_n as u64;

            // 命中：提升 LRU
            for &e in &plan.cache_hits {
                cache.touch(ExpertKey { layer: layer as u32, expert: e });
            }
            // Fetch：计入 PCIe 时间，并插入缓存
            for m in &plan.misses {
                if let MissAction::Fetch { expert_id } = m {
                    let ns = (bytes_per_expert / (profile.pcie_gbps * 1e9) * 1e9) as u64;
                    pcie_ns += ns;
                    cache.insert(ExpertKey { layer: layer as u32, expert: *expert_id });
                }
            }
            // CpuCompute：真实内核，只跑这些专家（其余 id = -1 跳过）
            let mut ids = vec![-1i32; tokens * k];
            let mut any_cpu = false;
            for (i, m) in plan.misses.iter().enumerate() {
                if let MissAction::CpuCompute { expert_id } = m {
                    ids[i] = *expert_id as i32;
                    any_cpu = true;
                }
            }
            if any_cpu {
                let t0 = Instant::now();
                #[cfg(feature = "cpu-simd")]
                {
                    ft_kernel::moe_bf16(
                        &mut h, &w13, &w2, &ids, &rw,
                        tokens, hidden, inter, experts, k, threads,
                    )
                    .expect("moe_bf16");
                }
                #[cfg(not(feature = "cpu-simd"))]
                {
                    let _ = (&w13, &w2, &h, threads);
                }
                cpu_ns += t0.elapsed().as_nanos() as u64;
            }
        }
    }
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let pcie_ms = pcie_ns as f64 / 1e6;
    let cpu_ms = cpu_ns as f64 / 1e6;
    // hybrid 重叠：PCIe fetch 与 CPU compute 并行，步时 ≈ max(pcie, cpu) + 命中(0)
    let overlap_ms = pcie_ms.max(cpu_ms);
    let tok_s = if overlap_ms > 0.0 { steps as f64 / (overlap_ms / 1000.0) } else { 0.0 };
    let wall_tok_s = if wall_ms > 0.0 { steps as f64 / (wall_ms / 1000.0) } else { 0.0 };

    let total_access = hits + fetches + cpu_misses;
    println!(
        "hybrid-smoke: {steps} tok × {layers} layers | cache {cache_slots} slots"
    );
    println!(
        "  routing: hits={hits} ({:.1}%) fetch={fetches} cpu={cpu_misses} / {total_access}",
        100.0 * hits as f64 / total_access.max(1) as f64
    );
    println!(
        "  time: wall={wall_ms:.0} ms  cpu={cpu_ms:.0} ms  pcie(model)={pcie_ms:.0} ms  overlap≈{overlap_ms:.0} ms"
    );
    println!(
        "  tok/s: wall={wall_tok_s:.1}  hybrid-overlap={tok_s:.1}  (FreeToken DSV4 实测 31–32)"
    );
    let _ = Duration::from_nanos(0);
}
