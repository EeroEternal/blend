//! 单专家 SwiGLU：GPU bf16 GEMV vs CPU naive，数值对拍。
pub fn run(hidden: usize, inter: usize) -> anyhow::Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (hidden, inter);
        anyhow::bail!("需要 --features cuda");
    }
    #[cfg(feature = "cuda")]
    {
        use ft_kernel::{device_count, expert_ffn, gpu_zero, set_device, DevBuffer, Stream};
        use ft_moe::{CpuMoeExecutor, NaiveF32Executor};

        let n = device_count()?;
        set_device((n as i32) - 1)?;
        let (h, i) = (hidden, inter);
        let mut rng = 0x1234u64;
        let mut rnd = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / u32::MAX as f32 - 0.5
        };
        let scale = 0.05f32 / (h as f32).sqrt();
        let w13: Vec<f32> = (0..2 * i * h).map(|_| rnd() * scale).collect();
        let w2: Vec<f32> = (0..h * i).map(|_| rnd() * scale).collect();
        let x: Vec<f32> = (0..h).map(|_| rnd()).collect();
        let f32_to_bf16 = |f: f32| -> u16 {
            let u = f.to_bits();
            ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
        };
        let w13b: Vec<u16> = w13.iter().copied().map(f32_to_bf16).collect();
        let w2b: Vec<u16> = w2.iter().copied().map(f32_to_bf16).collect();

        // CPU 参考（单专家，rw=1）
        let mut y_cpu = x.clone();
        NaiveF32Executor
            .forward(&mut y_cpu, 1, h, &w13, &w2, i, 1, &[0], &[1.0], 1)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // GPU：打包 slot = w13b | w2b
        let mut slot = w13b.clone();
        slot.extend_from_slice(&w2b);
        let mut d_slot = DevBuffer::alloc(slot.len() * 2)?;
        d_slot.h2d_bytes(slot.as_ptr() as *const _, slot.len() * 2)?;
        let mut d_x = DevBuffer::alloc(h * 4)?;
        d_x.h2d(&x)?;
        let mut d_y = DevBuffer::alloc(h * 4)?;
        let mut d_s2 = DevBuffer::alloc(2 * i * 4)?;
        let mut d_si = DevBuffer::alloc(i * 4)?;
        let stream = Stream::new()?;
        gpu_zero(&mut d_y, h, &stream)?;
        expert_ffn(
            d_slot.as_ptr() as *const u16,
            &d_x, &mut d_y, &mut d_s2, &mut d_si,
            h, i, 1.0, &stream,
        )?;
        stream.sync()?;
        let mut y_gpu = vec![0f32; h];
        d_y.d2h(&mut y_gpu)?;

        let max = y_cpu.iter().zip(y_gpu.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let gmax = y_cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        let rel = max / gmax;
        println!(
            "gpu-ffn-parity: H{h} I{i} | max_abs={max:.3e} rel={rel:.3e} cpu[0]={:.4} gpu[0]={:.4}",
            y_cpu[0], y_gpu[0]
        );
        if rel > 0.05 {
            anyhow::bail!("FAIL rel {rel:.3e} > 0.05");
        }
        println!("PASS");
        Ok(())
    }
}
