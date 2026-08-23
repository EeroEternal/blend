//! 数值对拍：读 fixture（torch 生成），跑 NaiveF32Executor，与 golden 比较。
//! fixture 布局：
//!   manifest.json : {"tokens":T,"hidden":H,"inter":I,"num_experts":E,"k":K}
//!   h_in.f32      : [T*H]   topk.f32 : [T*K](f32 存专家 id)   weights.f32 : [T*K]
//!   w13.f32       : [E*2*I*H]   w2.f32 : [E*H*I]            h_golden.f32 : [T*H]
use anyhow::{bail, Ok};
use ft_moe::{CpuMoeExecutor, NaiveF32Executor};
use std::path::Path;

/// f32 -> bf16 (round-to-nearest-even)，与 FreeToken/内核口径一致
fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    let lsb = (u >> 16) & 1;
    (((u + 0x7fff + lsb) >> 16) as u16)
}

fn read_bf16(p: &Path) -> anyhow::Result<Vec<u16>> {
    Ok(read_f32(p)?.iter().map(|&v| f32_to_bf16(v)).collect())
}

#[allow(dead_code)]
fn read_f32(p: &Path) -> anyhow::Result<Vec<f32>> {
    let raw = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

pub fn run(dir: &Path, tol: f32, kernel: &str) -> anyhow::Result<()> {
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let t = manifest["tokens"].as_u64().context("tokens")? as usize;
    let h = manifest["hidden"].as_u64().context("hidden")? as usize;
    let i = manifest["inter"].as_u64().context("inter")? as usize;
    let e = manifest["num_experts"].as_u64().context("num_experts")? as usize;
    let k = manifest["k"].as_u64().context("k")? as usize;

    let h_in = read_f32(&dir.join("h_in.f32"))?;
    let topk_f = read_f32(&dir.join("topk.f32"))?;
    let weights = read_f32(&dir.join("weights.f32"))?;
    let w13 = read_f32(&dir.join("w13.f32"))?;
    let w2 = read_f32(&dir.join("w2.f32"))?;
    let golden = read_f32(&dir.join("h_golden.f32"))?;
    let topk: Vec<u32> = topk_f.iter().map(|&v| v as u32).collect();

    assert_eq!(h_in.len(), t * h, "h_in size");
    assert_eq!(topk.len(), t * k, "topk size");
    assert_eq!(w13.len(), e * 2 * i * h, "w13 size");
    assert_eq!(w2.len(), e * h * i, "w2 size");

    let mut out = h_in.clone();
    let topk_i32: Vec<i32> = topk.iter().map(|&v| v as i32).collect();
    match kernel {
        "naive" => NaiveF32Executor
            .forward(&mut out, t, h, &w13, &w2, i, e, &topk, &weights, k)
            .map_err(|err| anyhow::anyhow!("executor: {err}"))?,
        #[cfg(feature = "cpu-simd")]
        "simd" => {
            // 权重转 bf16（对齐生产路径的数值口径）
            let _ = (&w13, &w2);
            ft_kernel::moe_bf16(
                &mut out,
                &read_bf16(&dir.join("w13.f32"))?,
                &read_bf16(&dir.join("w2.f32"))?,
                &topk_i32,
                &weights,
                t, h, i, e, k,
                ft_moe::physical_cores(),
            )?
        }
        other => anyhow::bail!("unknown kernel '{other}' (available: naive{})",
            if cfg!(feature = "cpu-simd") { ", simd" } else { "" }),
    }

    let max_diff = out
        .iter()
        .zip(golden.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // 相对误差：以 golden 最大幅值为归一化基准
    // （golden 幅值随权重缩放变化，绝对误差无跨形状可比性）
    let gmax = golden.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let rel = max_diff / gmax;
    println!(
        "parity[{kernel}]: tokens={t} hidden={h} inter={i} experts={e} k={k} | \
         max_abs={max_diff:.3e} golden_max={gmax:.3e} rel={rel:.3e} tol(rel)={tol:.3e}"
    );
    if rel <= tol {
        println!("PASS");
        Ok(())
    } else {
        bail!("FAIL: rel {rel:.3e} > tol {tol:.3e}")
    }
}
