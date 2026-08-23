//! Hybrid 路径冒烟：通过 ft-engine::DecodeDriver + ft-moe::HybridRuntime，
//! 内核走真实 SIMD moe_bf16。
use ft_engine::{DecodeDriver, MoeKernel};
use ft_moe::{BandwidthProfile, HybridRuntime, MissAction, MoePlan, QStarPolicy};
use std::time::Instant;

struct SimdKernel {
    w13: Vec<u16>,
    w2: Vec<u16>,
    h: Vec<f32>,
    rw: Vec<f32>,
    hidden: usize,
    inter: usize,
    experts: usize,
    k: usize,
    threads: usize,
    cpu_ns: u64,
    pcie_ns: u64,
    bytes_per_expert: f64,
    pcie_gbps: f64,
    counting: bool,
}

impl MoeKernel for SimdKernel {
    fn compute(&mut self, plan: &MoePlan, _layer: u32) {
        for m in &plan.misses {
            if let MissAction::Fetch { .. } = m {
                let ns = (self.bytes_per_expert / (self.pcie_gbps * 1e9) * 1e9) as u64;
                if self.counting {
                    self.pcie_ns += ns;
                }
            }
        }
        let ids = HybridRuntime::cpu_ids(plan, self.k);
        if ids.iter().all(|&e| e < 0) {
            return;
        }
        let t0 = Instant::now();
        #[cfg(feature = "cpu-simd")]
        {
            ft_kernel::moe_bf16(
                &mut self.h, &self.w13, &self.w2, &ids, &self.rw,
                1, self.hidden, self.inter, self.experts, self.k, self.threads,
            )
            .expect("moe_bf16");
        }
        if self.counting {
            self.cpu_ns += t0.elapsed().as_nanos() as u64;
        }
    }
}

pub fn run(steps: usize, layers: usize, cache_slots: usize, threads: usize) {
    let (hidden, inter, experts, k) = (4096usize, 2048usize, 256usize, 6usize);
    let threads = if threads == 0 { ft_moe::physical_cores() } else { threads };
    let profile = BandwidthProfile::pro6000_epyc9355();
    let q = QStarPolicy::calibrate(&profile);
    let bytes_per_expert = ((2 * inter * hidden + hidden * inter) * 2) as f64;

    eprintln!(
        "hybrid-smoke: steps={steps} layers={layers} cache={cache_slots} slots, \
         fetch_frac={:.1}%, threads={threads}",
        q.fetch_fraction() * 100.0
    );

    let mut rng = 0xA5A5_u64;
    let mut rnd = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
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
    let h: Vec<f32> = (0..hidden)
        .map(|_| (rnd() >> 40) as f32 / u32::MAX as f32 - 0.5)
        .collect();
    let rw: Vec<f32> = vec![1.0 / k as f32; k];

    let kernel = SimdKernel {
        w13, w2, h, rw, hidden, inter, experts, k, threads,
        cpu_ns: 0, pcie_ns: 0, bytes_per_expert, pcie_gbps: profile.pcie_gbps,
        counting: false,
    };
    let mut drv = DecodeDriver::new(cache_slots, q, kernel, layers);

    // 每层上一 token 的路由（时间局部性）
    let mut prev: Vec<Vec<u32>> = (0..layers)
        .map(|layer| {
            (0..k)
                .map(|i| ((layer * 17 + i * 31) % experts) as u32)
                .collect()
        })
        .collect();

    let warmup = steps / 2;
    let wall = Instant::now();
    for step in 0..steps {
        drv.kernel.counting = step >= warmup;
        if step == warmup {
            drv.rt.reset_stats();
        }
        drv.decode_token(|layer| {
            let l = layer as usize;
            if step > 0 {
                prev[l][k - 1] = ((l * 13 + step * 7) % experts) as u32;
            }
            prev[l].clone()
        });
    }
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let cpu_ms = drv.kernel.cpu_ns as f64 / 1e6;
    let pcie_ms = drv.kernel.pcie_ns as f64 / 1e6;
    let overlap_ms = pcie_ms.max(cpu_ms);
    let measured = (steps - warmup).max(1) as f64;
    let tok_s = if overlap_ms > 0.0 { measured / (overlap_ms / 1000.0) } else { 0.0 };
    let wall_tok_s = steps as f64 / (wall_ms / 1000.0);
    let st = drv.rt.stats();

    println!("hybrid-smoke: {steps} tok × {layers} layers | cache {cache_slots} slots");
    println!(
        "  routing: hits={} ({:.1}%) fetch={} cpu={} / {}",
        st.hits,
        100.0 * st.hit_rate(),
        st.fetches,
        st.cpu_misses,
        st.total()
    );
    println!(
        "  time: wall={wall_ms:.0} ms  cpu={cpu_ms:.0} ms  pcie(model)={pcie_ms:.0} ms  overlap≈{overlap_ms:.0} ms"
    );
    println!(
        "  tok/s: wall={wall_tok_s:.1}  hybrid-overlap={tok_s:.1}  (FreeToken DSV4 实测 31–32)"
    );
}
