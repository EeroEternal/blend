//! 从真实 HF 目录加载一个 Qwen3-MoE 专家，CPU vs GPU 对拍。
use anyhow::{bail, Context};
use ft_loader::{locate_tensor, SafeTensorFile};
use std::path::Path;

pub fn run(model: &str, layer: usize, expert: usize) -> anyhow::Result<()> {
    let root = Path::new(model);
    let names = [
        format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
        format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
        format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
    ];
    let mut tensors = Vec::new();
    for n in &names {
        let shard = locate_tensor(root, n)?;
        let st = SafeTensorFile::open(&shard)?;
        let tv = st.tensor(n).ok_or_else(|| anyhow::anyhow!("missing {n} in {}", shard.display()))?;
        eprintln!("  {} {:?} {}", n, tv.shape, tv.dtype);
        tensors.push((tv.dtype.to_string(), tv.shape.to_vec(), tv.as_u16().or_else(|_| {
            // F32 权重转 bf16
            tv.as_f32().map(|f| {
                f.iter().map(|&x| {
                    let u = x.to_bits();
                    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
                }).collect()
            })
        }).with_context(|| format!("decode {n}"))?));
    }
    let (gate_d, gate_s, gate) = &tensors[0];
    let (_up_d, up_s, up) = &tensors[1];
    let (_dn_d, dn_s, down) = &tensors[2];
    let _ = gate_d;
    // gate [I,H], up [I,H], down [H,I]
    let inter = gate_s[0];
    let hidden = gate_s[1];
    if up_s != gate_s || dn_s[0] != hidden || dn_s[1] != inter {
        bail!("shape mismatch gate={gate_s:?} up={up_s:?} down={dn_s:?}");
    }
    let mut w13 = gate.clone();
    w13.extend_from_slice(up);
    let w2 = down.clone();

    // 随机激活
    let mut rng = 0x51u64;
    let x: Vec<f32> = (0..hidden)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / u32::MAX as f32 * 0.1
        })
        .collect();
    let w13_f: Vec<f32> = w13.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    let w2_f: Vec<f32> = w2.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();

    use ft_moe::{CpuMoeExecutor, NaiveF32Executor};
    let mut y_cpu = x.clone();
    NaiveF32Executor
        .forward(&mut y_cpu, 1, hidden, &w13_f, &w2_f, inter, 1, &[0], &[1.0], 1)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    #[cfg(feature = "cpu-simd")]
    {
        let mut y_simd = x.clone();
        ft_kernel::moe_bf16(&mut y_simd, &w13, &w2, &[0], &[1.0], 1, hidden, inter, 1, 1, 1)?;
        let rel = rel_err(&y_cpu, &y_simd);
        println!("real-expert cpu-simd vs naive: rel={rel:.3e}");
    }

    #[cfg(feature = "cuda")]
    {
        use ft_kernel::{device_count, expert_ffn, gpu_zero, set_device, DevBuffer, Stream};
        let n = device_count()?;
        set_device((n as i32) - 1)?;
        let mut slot = w13.clone();
        slot.extend_from_slice(&w2);
        let mut d_slot = DevBuffer::alloc(slot.len() * 2)?;
        d_slot.h2d_bytes(slot.as_ptr() as *const _, slot.len() * 2)?;
        let mut d_x = DevBuffer::alloc(hidden * 4)?;
        d_x.h2d(&x)?;
        let mut d_y = DevBuffer::alloc(hidden * 4)?;
        let mut d_s2 = DevBuffer::alloc(2 * inter * 4)?;
        let mut d_si = DevBuffer::alloc(inter * 4)?;
        let stream = Stream::new()?;
        gpu_zero(&mut d_y, hidden, &stream)?;
        expert_ffn(
            d_slot.as_ptr() as *const u16,
            &d_x, &mut d_y, &mut d_s2, &mut d_si,
            hidden, inter, 1.0, &stream,
        )?;
        stream.sync()?;
        let mut y_gpu = vec![0f32; hidden];
        d_y.d2h(&mut y_gpu)?;
        let rel = rel_err(&y_cpu, &y_gpu);
        println!(
            "real-expert GPU vs naive: L{layer} E{expert} H{hidden} I{inter} rel={rel:.3e} y[0] cpu={:.4} gpu={:.4}",
            y_cpu[0], y_gpu[0]
        );
        if rel > 0.05 {
            bail!("FAIL rel {rel:.3e}");
        }
        println!("PASS");
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("real-expert loaded L{layer} E{expert} H{hidden} I{inter} (no cuda, skip GPU)");
    }
    Ok(())
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let max = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let g = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    max / g
}
