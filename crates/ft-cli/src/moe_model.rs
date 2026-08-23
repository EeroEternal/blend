//! 加载 Qwen3-MoE 全部层的专家，串行 decode（attention stub = 残差直通）。
use anyhow::Context;
use ft_loader::{locate_tensor, SafeTensorFile};
use std::collections::HashMap;
use std::path::Path;

struct Layer {
    gate: Vec<u16>, // [E, H]
    w13: Vec<u16>,  // [E, 2I, H]
    w2: Vec<u16>,   // [E, H, I]
}

pub fn run(model: &str, n_layers: usize, steps: usize) -> anyhow::Result<()> {
    let root = Path::new(model);
    let g0 = load_u16(root, "model.layers.0.mlp.gate.weight")?;
    let e = g0.shape[0];
    let hidden = g0.shape[1];
    let gp0 = load_u16(root, "model.layers.0.mlp.experts.0.gate_proj.weight")?;
    let inter = gp0.shape[0];
    let k = 8usize;
    let n_layers = n_layers.min(64);
    eprintln!("moe-model: {n_layers} layers H{hidden} I{inter} E{e} k{k}");

    // 收集 (shard -> [(layer, kind, expert_or_gate, name)])
    #[derive(Clone)]
    enum Kind {
        Gate,
        ExpertGate(usize),
        ExpertUp(usize),
        ExpertDown(usize),
    }
    let mut by_shard: HashMap<String, Vec<(usize, Kind, String)>> = HashMap::new();
    for layer in 0..n_layers {
        let gn = format!("model.layers.{layer}.mlp.gate.weight");
        let shard = locate_tensor(root, &gn)?;
        by_shard.entry(shard.display().to_string()).or_default().push((layer, Kind::Gate, gn));
        for ei in 0..e {
            for (kind, suf) in [
                (Kind::ExpertGate(ei), "gate_proj"),
                (Kind::ExpertUp(ei), "up_proj"),
                (Kind::ExpertDown(ei), "down_proj"),
            ] {
                let name = format!("model.layers.{layer}.mlp.experts.{ei}.{suf}.weight");
                let shard = locate_tensor(root, &name)?;
                by_shard.entry(shard.display().to_string()).or_default().push((layer, kind.clone(), name));
            }
        }
    }

    let mut layers: Vec<Layer> = (0..n_layers)
        .map(|_| Layer {
            gate: vec![0; e * hidden],
            w13: vec![0; e * 2 * inter * hidden],
            w2: vec![0; e * hidden * inter],
        })
        .collect();

    let t_load = std::time::Instant::now();
    for (shard, items) in &by_shard {
        let st = SafeTensorFile::open(Path::new(shard))?;
        for (layer, kind, name) in items {
            let tv = st.tensor(name).ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
            let bits = tv.as_u16().with_context(|| name.clone())?;
            let L = &mut layers[*layer];
            match kind {
                Kind::Gate => L.gate.copy_from_slice(&bits),
                Kind::ExpertGate(ei) => {
                    let off = *ei * 2 * inter * hidden;
                    L.w13[off..off + inter * hidden].copy_from_slice(&bits);
                }
                Kind::ExpertUp(ei) => {
                    let off = *ei * 2 * inter * hidden + inter * hidden;
                    L.w13[off..off + inter * hidden].copy_from_slice(&bits);
                }
                Kind::ExpertDown(ei) => {
                    let off = *ei * hidden * inter;
                    L.w2[off..off + hidden * inter].copy_from_slice(&bits);
                }
            }
        }
    }
    let gb = n_layers as f64 * (e * (3 * inter * hidden) * 2) as f64 / 1e9;
    eprintln!("  loaded in {:.1}s ({gb:.1} GB packed)", t_load.elapsed().as_secs_f64());

    let mut rng = 0x1111u64;
    let mut x: Vec<f32> = (0..hidden)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / u32::MAX as f32 * 0.02
        })
        .collect();

    #[cfg(feature = "cpu-simd")]
    {
        let th = ft_moe::physical_cores();
        // warmup
        one_step(&mut x, &layers, hidden, inter, e, k, th)?;
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            one_step(&mut x, &layers, hidden, inter, e, k, th)?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps.max(1) as f64;
        println!(
            "moe-model: {n_layers} layers × {steps} tok: {:.1} ms/tok ({:.1} tok/s)  |x|={:.4}",
            ms,
            1000.0 / ms,
            x.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        );
    }
    #[cfg(not(feature = "cpu-simd"))]
    {
        let _ = (steps, &mut x, &layers, hidden, inter, e, k);
        println!("moe-model loaded, rebuild with --features cpu-simd to run");
    }
    Ok(())
}

#[cfg(feature = "cpu-simd")]
fn one_step(
    x: &mut [f32],
    layers: &[Layer],
    hidden: usize,
    inter: usize,
    e: usize,
    k: usize,
    th: usize,
) -> anyhow::Result<()> {
    for L in layers {
        let (ids, rw) = route(&L.gate, x, e, hidden, k);
        let ids_i: Vec<i32> = ids.iter().map(|&v| v as i32).collect();
        // 残差：先算 MoE 到 scratch，再加回 x
        let mut y = x.to_vec();
        ft_kernel::moe_bf16(&mut y, &L.w13, &L.w2, &ids_i, &rw, 1, hidden, inter, e, k, th)?;
        for (a, b) in x.iter_mut().zip(y.iter()) {
            *a += *b;
        }
    }
    Ok(())
}

fn route(gate: &[u16], x: &[f32], e: usize, hidden: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut logits = vec![0f32; e];
    for ei in 0..e {
        let mut s = 0f32;
        for j in 0..hidden {
            s += f32::from_bits((gate[ei * hidden + j] as u32) << 16) * x[j];
        }
        logits[ei] = s;
    }
    let mut order: Vec<usize> = (0..e).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let maxl = order[..k].iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut w = vec![0f32; k];
    let mut sum = 0f32;
    for (j, &ei) in order[..k].iter().enumerate() {
        w[j] = (logits[ei] - maxl).exp();
        sum += w[j];
    }
    let ids: Vec<u32> = order[..k].iter().map(|&i| i as u32).collect();
    let rw: Vec<f32> = w.iter().map(|v| v / sum).collect();
    (ids, rw)
}

struct Loaded {
    shape: Vec<usize>,
}

fn load_u16(root: &Path, name: &str) -> anyhow::Result<Loaded> {
    let shard = locate_tensor(root, name)?;
    let st = SafeTensorFile::open(&shard)?;
    let tv = st.tensor(name).ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
    let _ = tv.as_u16()?;
    Ok(Loaded { shape: tv.shape.to_vec() })
}
