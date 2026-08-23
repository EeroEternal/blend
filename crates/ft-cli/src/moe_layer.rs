//! 加载 Qwen3-MoE 一整层（router + 全部专家），真实 top-k 前向并对拍。
use anyhow::{bail, Context};
use ft_loader::{locate_tensor, SafeTensorFile};
use ft_moe::{CpuMoeExecutor, NaiveF32Executor};
use std::collections::HashMap;
use std::path::Path;

pub fn run(model: &str, layer: usize, tokens: usize) -> anyhow::Result<()> {
    let root = Path::new(model);
    let gate_name = format!("model.layers.{layer}.mlp.gate.weight");
    let gate = load_u16(root, &gate_name)?;
    let e = gate.shape[0];
    let hidden = gate.shape[1];
    eprintln!("router {gate_name} {:?}", gate.shape);

    // 先读 expert 0 定 I
    let g0 = load_u16(root, &format!("model.layers.{layer}.mlp.experts.0.gate_proj.weight"))?;
    let inter = g0.shape[0];
    assert_eq!(g0.shape[1], hidden);

    eprintln!("loading {e} experts H{hidden} I{inter} ...");
    let mut w13 = vec![0u16; e * 2 * inter * hidden];
    let mut w2 = vec![0u16; e * hidden * inter];
    // 按分片批量打开，避免 384 次 open
    let mut needed: HashMap<String, Vec<(usize, &'static str, String)>> = HashMap::new();
    for ei in 0..e {
        for (kind, suffix) in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")] {
            let name = format!("model.layers.{layer}.mlp.experts.{ei}.{suffix}.weight");
            let shard = locate_tensor(root, &name)?;
            needed.entry(shard.display().to_string()).or_default().push((
                ei,
                kind,
                name,
            ));
        }
    }
    for (shard, items) in &needed {
        let st = SafeTensorFile::open(Path::new(shard))?;
        for (ei, kind, name) in items {
            let tv = st.tensor(name).ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
            let bits = tv.as_u16().with_context(|| name.clone())?;
            match *kind {
                "gate" => {
                    let off = *ei * 2 * inter * hidden;
                    w13[off..off + inter * hidden].copy_from_slice(&bits);
                }
                "up" => {
                    let off = *ei * 2 * inter * hidden + inter * hidden;
                    w13[off..off + inter * hidden].copy_from_slice(&bits);
                }
                "down" => {
                    let off = *ei * hidden * inter;
                    w2[off..off + hidden * inter].copy_from_slice(&bits);
                }
                _ => {}
            }
        }
    }
    eprintln!("  packed w13 {} MB  w2 {} MB", w13.len() * 2 / 1_000_000, w2.len() * 2 / 1_000_000);

    // 随机 token 激活
    let mut rng = 0xC0FFEEu64;
    let mut rnd = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 40) as f32 / u32::MAX as f32 * 0.05
    };
    let x: Vec<f32> = (0..tokens * hidden).map(|_| rnd()).collect();
    let k = 8usize;

    // 真实 router：logits = gate @ x，每 token top-k + softmax 归一化
    let gate_f: Vec<f32> = gate.bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    let mut ids = vec![0u32; tokens * k];
    let mut rw = vec![0f32; tokens * k];
    for t in 0..tokens {
        let xt = &x[t * hidden..(t + 1) * hidden];
        let mut logits = vec![0f32; e];
        for ei in 0..e {
            let row = &gate_f[ei * hidden..(ei + 1) * hidden];
            logits[ei] = row.iter().zip(xt.iter()).map(|(a, b)| a * b).sum();
        }
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let top = &order[..k];
        let maxl = top.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        let mut w = vec![0f32; k];
        for (j, &ei) in top.iter().enumerate() {
            w[j] = (logits[ei] - maxl).exp();
            sum += w[j];
        }
        for j in 0..k {
            ids[t * k + j] = top[j] as u32;
            rw[t * k + j] = w[j] / sum;
        }
    }

    let w13_f: Vec<f32> = w13.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    let w2_f: Vec<f32> = w2.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();

    let mut y_naive = x.clone();
    let t0 = std::time::Instant::now();
    NaiveF32Executor
        .forward(&mut y_naive, tokens, hidden, &w13_f, &w2_f, inter, e, &ids, &rw, k)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    println!("moe-layer naive: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);

    #[cfg(feature = "cpu-simd")]
    {
        let mut y = x.clone();
        let ids_i: Vec<i32> = ids.iter().map(|&v| v as i32).collect();
        let th = ft_moe::physical_cores();
        let t0 = std::time::Instant::now();
        ft_kernel::moe_bf16(&mut y, &w13, &w2, &ids_i, &rw, tokens, hidden, inter, e, k, th)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rel = max_rel(&y_naive, &y);
        println!("moe-layer simd:  {ms:.1} ms  vs naive rel={rel:.3e}  (L{layer} T{tokens} E{e} k{k})");
        if rel > 0.05 {
            bail!("FAIL rel {rel:.3e}");
        }
        println!("PASS");
    }
    #[cfg(not(feature = "cpu-simd"))]
    {
        println!("moe-layer loaded (no cpu-simd)");
    }
    Ok(())
}

struct Loaded {
    shape: Vec<usize>,
    bits: Vec<u16>,
}

fn load_u16(root: &Path, name: &str) -> anyhow::Result<Loaded> {
    let shard = locate_tensor(root, name)?;
    let st = SafeTensorFile::open(&shard)?;
    let tv = st.tensor(name).ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
    Ok(Loaded { shape: tv.shape.to_vec(), bits: tv.as_u16()? })
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let m = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    m / a.iter().fold(0.0f32, |acc, v| acc.max(v.abs())).max(1e-30)
}
