//! GPU 链路冒烟：设备查询 → H2D → ft_vector_add → D2H → 校验。
//! 验证 Rust → FFI → libftkernels.so → CUDA 全链路（P2 第一个真机里程碑）。
use anyhow::{bail, Context, Result};

pub fn run(n: usize) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = n;
        bail!("本二进制未启用 cuda feature；请用: cargo build --release --features cuda (需 FT_KERNELS_DIR 指向 libftkernels.so)");
    }

    #[cfg(feature = "cuda")]
    {
        use ft_kernel::{device_count, device_info, vector_add, DevBuffer};

        let count = device_count().context("device_count")?;
        println!("CUDA devices: {count}");
        if count == 0 {
            bail!("no CUDA device");
        }
        for i in 0..count.min(8) {
            let info = device_info(i as i32)?;
            println!("  [{i}] {}", info.name);
        }
        // 用最后一块卡（6000 Pro 机器上 4/5 空闲，避免打扰业务）
        let dev = count as i32 - 1;
        let info = device_info(dev)?;
        println!("using device [{dev}] {}", info.name);

        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

        let da = DevBuffer::alloc(n * 4)?;
        let db = DevBuffer::alloc(n * 4)?;
        let mut dout = DevBuffer::alloc(n * 4)?;
        da.h2d(&a)?;
        db.h2d(&b)?;
        vector_add(&da, &db, &mut dout, n)?;

        let mut out = vec![0f32; n];
        dout.d2h(&mut out)?;

        let mismatches: usize = out
            .iter()
            .enumerate()
            .filter(|&(i, v)| (*v - (a[i] + b[i])).abs() > 1e-5)
            .count();
        if mismatches == 0 {
            println!("gpu-smoke PASS: {n} elements, out[i] == a[i]+b[i]");
            Ok(())
        } else {
            bail!("gpu-smoke FAIL: {mismatches}/{n} mismatched");
        }
    }
}
