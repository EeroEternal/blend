//! Hybrid 路径冒烟：通过 ft-engine::DecodeDriver + ft-moe::HybridRuntime，
//! 内核走真实 SIMD moe_bf16。
use ft_engine::{DecodeDriver, MoeKernel};
use ft_moe::{BandwidthProfile, HybridRuntime, MissAction, MoePlan, QStarPolicy};
use std::time::Instant;

#[cfg(feature = "cuda")]
fn init_gpu_ctx(slot_bytes: usize, hidden: usize, inter: usize) -> anyhow::Result<GpuCtx> {
    use ft_kernel::{device_count, set_device, DevBuffer, GpuSlotBank, Stream};
    let n = device_count()?;
    if n == 0 {
        anyhow::bail!("no CUDA device");
    }
    set_device((n as i32) - 1)?;
    Ok(GpuCtx {
        bank: GpuSlotBank::new(64, slot_bytes)?,
        stream: Stream::new()?,
        x: DevBuffer::alloc(hidden * 4)?,
        y: DevBuffer::alloc(hidden * 4)?,
        s2: DevBuffer::alloc(2 * inter * 4)?,
        si: DevBuffer::alloc(inter * 4)?,
    })
}

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
    real_pcie_ns: u64,
    bytes_per_expert: f64,
    pcie_gbps: f64,
    counting: bool,
    real_pcie: bool,
    next_slot: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuCtx>,
}

#[cfg(feature = "cuda")]
struct GpuCtx {
    bank: ft_kernel::GpuSlotBank,
    stream: ft_kernel::Stream,
    x: ft_kernel::DevBuffer,
    y: ft_kernel::DevBuffer,
    s2: ft_kernel::DevBuffer,
    si: ft_kernel::DevBuffer,
}

impl MoeKernel for SimdKernel {
    fn compute(&mut self, plan: &MoePlan, _layer: u32) {
        let mut n_fetch = 0usize;
        for m in &plan.misses {
            if let MissAction::Fetch { expert_id } = m {
                let ns = (self.bytes_per_expert / (self.pcie_gbps * 1e9) * 1e9) as u64;
                if self.counting {
                    self.pcie_ns += ns;
                }
                n_fetch += 1;
                let _ = expert_id;
                #[cfg(feature = "cuda")]
                if let Some(ctx) = self.gpu.as_mut() {
                    let e = *expert_id as usize;
                    upload_expert(ctx, &self.w13, &self.w2, e, self.hidden, self.inter, &mut self.next_slot);
                }
            }
        }
        let _ = n_fetch;
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
        #[cfg(feature = "cuda")]
        if self.real_pcie {
            if let Some(ctx) = self.gpu.as_mut() {
                let t1 = Instant::now();
                // 命中 + 本步 Fetch 的专家在 GPU 上算，累加到 y
                let _ = ft_kernel::gpu_zero(&mut ctx.y, self.hidden, &ctx.stream);
                let _ = ctx.x.h2d(&self.h);
                let gpu_experts: Vec<u32> = plan
                    .cache_hits
                    .iter()
                    .copied()
                    .chain(plan.misses.iter().filter_map(|m| match m {
                        MissAction::Fetch { expert_id } => Some(*expert_id),
                        _ => None,
                    }))
                    .collect();
                for e in gpu_experts {
                    let slot = (e as usize) % ctx.bank.slots;
                    let _ = ft_kernel::expert_ffn(
                        ctx.bank.slot_ptr(slot) as *const u16,
                        &ctx.x, &mut ctx.y, &mut ctx.s2, &mut ctx.si,
                        self.hidden, self.inter, 1.0 / self.k as f32, &ctx.stream,
                    );
                }
                let _ = ctx.stream.sync();
                if self.counting {
                    self.real_pcie_ns += t1.elapsed().as_nanos() as u64;
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn upload_expert(
    ctx: &GpuCtx,
    w13: &[u16],
    w2: &[u16],
    e: usize,
    hidden: usize,
    inter: usize,
    next_slot: &mut usize,
) {
    let w13_off = e * 2 * inter * hidden;
    let w2_off = e * hidden * inter;
    let w13_bytes = 2 * inter * hidden * 2;
    let slot = *next_slot % ctx.bank.slots;
    *next_slot += 1;
    let dst = ctx.bank.slot_ptr(slot);
    let _ = ctx.stream.h2d_async(dst, unsafe { w13.as_ptr().add(w13_off) } as *const _, w13_bytes);
    let _ = ctx.stream.h2d_async(
        unsafe { (dst as *mut u8).add(w13_bytes) } as *mut _,
        unsafe { w2.as_ptr().add(w2_off) } as *const _,
        hidden * inter * 2,
    );
}

pub fn run(steps: usize, layers: usize, cache_slots: usize, threads: usize, real_pcie: bool) {
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

    if real_pcie {
        eprintln!("  real-pcie: ON (cudaMemcpyAsync)");
    }
    let kernel = SimdKernel {
        w13, w2, h, rw, hidden, inter, experts, k, threads,
        cpu_ns: 0, pcie_ns: 0, real_pcie_ns: 0, bytes_per_expert, pcie_gbps: profile.pcie_gbps,
        counting: false, real_pcie, next_slot: 0,
        #[cfg(feature = "cuda")]
        gpu: if real_pcie {
            match init_gpu_ctx(bytes_per_expert as usize, hidden, inter) {
                Ok(g) => Some(g),
                Err(e) => {
                    eprintln!("  real-pcie init failed: {e}; falling back to modeled");
                    None
                }
            }
        } else {
            None
        },
    };
    let mut drv = DecodeDriver::new(cache_slots, q, kernel, layers);
    #[cfg(feature = "cuda")]
    if let Some(ctx) = drv.kernel.gpu.as_ref() {
        let bytes = drv.kernel.bytes_per_expert as usize;
        let t0 = Instant::now();
        let _ = ctx.stream.h2d_async(ctx.bank.slot_ptr(0), drv.kernel.w13.as_ptr() as *const _, bytes.min(drv.kernel.w13.len() * 2));
        let _ = ctx.stream.sync();
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "  expert H2D probe: {:.2} ms, {:.1} GB/s",
            dt * 1000.0,
            bytes as f64 / dt / 1e9
        );
    }

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
    let real_pcie_ms = drv.kernel.real_pcie_ns as f64 / 1e6;
    println!(
        "  time: wall={wall_ms:.0} ms  cpu={cpu_ms:.0} ms  pcie(model)={pcie_ms:.0} ms  pcie(real)={real_pcie_ms:.0} ms  overlap≈{overlap_ms:.0} ms"
    );
    println!(
        "  tok/s: wall={wall_tok_s:.1}  hybrid-overlap={tok_s:.1}  (FreeToken DSV4 实测 31–32)"
    );
}
