//! 实测 PCIe H2D 带宽（cudaMemcpyAsync + stream sync）。
//! 对照 FreeToken benchbw 的 57.7 GB/s。
pub fn run(mib: usize, iters: usize) -> anyhow::Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (mib, iters);
        anyhow::bail!("需要 --features cuda 构建");
    }
    #[cfg(feature = "cuda")]
    {
        use ft_kernel::{device_count, device_info, set_device, DevBuffer, PinnedBuf, Stream};
        let n = device_count()?;
        if n == 0 {
            anyhow::bail!("no CUDA device");
        }
        let dev = (n as i32) - 1;
        set_device(dev)?;
        let info = device_info(dev)?;
        println!("pcie-bench: device [{dev}] {}", info.name);

        let bytes = mib * 1024 * 1024;
        let host = PinnedBuf::alloc(bytes)?;
        let dst = DevBuffer::alloc(bytes)?;
        let stream = Stream::new()?;
        stream.h2d_async(dst.as_ptr() as *mut _, host.as_ptr(), bytes)?;
        stream.sync()?;

        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            stream.h2d_async(dst.as_ptr() as *mut _, host.as_ptr(), bytes)?;
            stream.sync()?;
            let dt = t0.elapsed().as_secs_f64();
            best = best.min(dt);
        }
        let gbps = bytes as f64 / best / 1e9;
        println!("pcie-bench: {mib} MiB x {iters}: best {:.2} ms, {:.1} GB/s (profile 57.7)", best * 1000.0, gbps);
        Ok(())
    }
}
