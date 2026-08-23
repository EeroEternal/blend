//! Qwen3-MoE 真实 decode：RMSNorm + GQA 注意力 + 专家 MoE。
//! 注意力走 CPU f32（decode bs=1、短序列够用）；MoE 走 AVX512BF16。
use anyhow::Context;
use ft_loader::{locate_tensor, SafeTensorFile};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "cuda")]
struct GpuAttn {
    stream: ft_kernel::Stream,
    x: ft_kernel::DevBuffer,
    y: ft_kernel::DevBuffer,
    /// 每层 4 块：q k v o
    w: Vec<ft_kernel::DevBuffer>,
}

#[cfg(feature = "cuda")]
impl GpuAttn {
    fn gemv(&mut self, layer: usize, which: usize, x: &[f32], rows: usize, cols: usize) -> anyhow::Result<Vec<f32>> {
        self.x.h2d(x)?;
        ft_kernel::gemv_bf16(
            self.w[layer * 4 + which].as_ptr() as *const u16,
            &self.x,
            &mut self.y,
            rows,
            cols,
            &self.stream,
        )?;
        self.stream.sync()?;
        let mut out = vec![0f32; rows];
        self.y.d2h(&mut out)?;
        Ok(out)
    }
}

const HEADS: usize = 32;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10_000_000.0;

struct Attn {
    q: Vec<u16>,  // [H_q, H] = [4096, 2048]
    k: Vec<u16>,
    v: Vec<u16>,
    o: Vec<u16>,  // [H, H_q]
    qn: Vec<f32>, // [head_dim]
    kn: Vec<f32>,
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
}

struct Layer {
    attn: Attn,
    gate: Vec<u16>,
    w13: Vec<u16>,
    w2: Vec<u16>,
}

pub fn run(model: &str, n_layers: usize, prompt_len: usize, steps: usize) -> anyhow::Result<()> {
    let root = Path::new(model);
    let n_layers = n_layers.min(48);
    let hidden = 2048usize;
    let e = 128usize;
    let inter = 768usize;
    let k_moe = 8usize;
    eprintln!("decode-qwen: {n_layers} layers, prompt={prompt_len} decode={steps}");

    let mut layers = load_layers(root, n_layers, hidden, e, inter)?;
    let final_norm = load_f32_from_bf16(root, "model.norm.weight")?;

    #[cfg(feature = "cuda")]
    let mut gpu_attn = match init_gpu_attn(&layers, hidden) {
        Ok(g) => {
            eprintln!("  GPU attn GEMV: on");
            Some(g)
        }
        Err(e) => {
            eprintln!("  GPU attn off ({e})");
            None
        }
    };
    #[cfg(not(feature = "cuda"))]
    let mut gpu_attn: Option<()> = None;
    let _ = &mut gpu_attn;

    // 用 embed 的前 prompt_len 个词作为 prompt（确定性、无需 tokenizer）
    let embed = load_u16(root, "model.embed_tokens.weight")?;
    let vocab = embed.shape[0];
    eprintln!("  embed vocab={vocab}  attn+moe loaded");

    let mut x: Vec<Vec<f32>> = Vec::new(); // prefill 各位置 hidden
    for t in 0..prompt_len {
        let id = (t * 17 + 3) % vocab.min(10000);
        let row = &embed.bits[id * hidden..(id + 1) * hidden];
        x.push(row.iter().map(|&b| bf16(b)).collect());
    }

    // KV cache: [layer][kv_head][seq][dim]
    let mut ck: Vec<Vec<Vec<[f32; HEAD_DIM]>>> = vec![vec![Vec::new(); KV_HEADS]; n_layers];
    let mut cv: Vec<Vec<Vec<[f32; HEAD_DIM]>>> = vec![vec![Vec::new(); KV_HEADS]; n_layers];

    let th = {
        #[cfg(feature = "cpu-simd")]
        { ft_moe::physical_cores() }
        #[cfg(not(feature = "cpu-simd"))]
        { 1 }
    };

    let t_pre = std::time::Instant::now();
    for pos in 0..prompt_len {
        let mut h = x[pos].clone();
        for li in 0..n_layers {
            layer_step(&mut h, pos, &layers[li], hidden, e, inter, k_moe, th, &mut ck[li], &mut cv[li], gpu_attn.as_mut(), li)?;
        }
        x[pos] = h;
    }
    eprintln!("  prefill {prompt_len} tok: {:.1} ms", t_pre.elapsed().as_secs_f64() * 1000.0);

    let mut h = x.last().cloned().unwrap();
    let t0 = std::time::Instant::now();
    for s in 0..steps {
        let pos = prompt_len + s;
        for li in 0..n_layers {
            layer_step(&mut h, pos, &layers[li], hidden, e, inter, k_moe, th, &mut ck[li], &mut cv[li], gpu_attn.as_mut(), li)?;
        }
        // 最终 norm（不跑 lm_head，只保数值不炸）
        rmsnorm_inplace(&mut h, &final_norm);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps.max(1) as f64;
    let mx = h.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!(
        "decode-qwen: {n_layers}L prompt={prompt_len} + {steps} gen: {:.1} ms/tok ({:.1} tok/s)  |h|={mx:.3}",
        ms,
        1000.0 / ms
    );
    let _ = &mut layers;
    Ok(())
}

fn layer_step(
    h: &mut [f32],
    pos: usize,
    layer: &Layer,
    hidden: usize,
    e: usize,
    inter: usize,
    k_moe: usize,
    th: usize,
    ck: &mut [Vec<[f32; HEAD_DIM]>],
    cv: &mut [Vec<[f32; HEAD_DIM]>],
    #[cfg(feature = "cuda")] gpu: Option<&mut GpuAttn>,
    #[cfg(not(feature = "cuda"))] gpu: Option<&mut ()>,
    layer_id: usize,
) -> anyhow::Result<()> {
    let mut n = h.to_vec();
    rmsnorm_inplace(&mut n, &layer.attn.in_norm);
    let attn_out = attention(&n, pos, &layer.attn, ck, cv, gpu, layer_id);
    for (a, b) in h.iter_mut().zip(attn_out.iter()) {
        *a += *b;
    }
    let mut n2 = h.to_vec();
    rmsnorm_inplace(&mut n2, &layer.attn.post_norm);
    moe_residual(h, &n2, layer, hidden, e, inter, k_moe, th)?;
    Ok(())
}

fn attention(
    x: &[f32],
    pos: usize,
    a: &Attn,
    ck: &mut [Vec<[f32; HEAD_DIM]>],
    cv: &mut [Vec<[f32; HEAD_DIM]>],
    #[cfg(feature = "cuda")] mut gpu: Option<&mut GpuAttn>,
    #[cfg(not(feature = "cuda"))] _gpu: Option<&mut ()>,
    layer_id: usize,
) -> Vec<f32> {
    let hq = HEADS * HEAD_DIM;
    let hk = KV_HEADS * HEAD_DIM;
    let mut gemv_w = |w: &[u16], x: &[f32], rows: usize, cols: usize, which: usize| -> Vec<f32> {
        #[cfg(feature = "cuda")]
        if let Some(g) = gpu.as_mut() {
            if let Ok(v) = g.gemv(layer_id, which, x, rows, cols) {
                return v;
            }
        }
        gemv(w, x, rows, cols)
    };
    let q = gemv_w(&a.q, x, hq, x.len(), 0);
    let k = gemv_w(&a.k, x, hk, x.len(), 1);
    let v = gemv_w(&a.v, x, hk, x.len(), 2);
    let mut qh = vec![[0f32; HEAD_DIM]; HEADS];
    let mut kh = vec![[0f32; HEAD_DIM]; KV_HEADS];
    let mut vh = vec![[0f32; HEAD_DIM]; KV_HEADS];
    for hd in 0..HEADS {
        for d in 0..HEAD_DIM {
            qh[hd][d] = q[hd * HEAD_DIM + d];
        }
        rmsnorm_head(&mut qh[hd], &a.qn);
        rope(&mut qh[hd], pos);
    }
    for hd in 0..KV_HEADS {
        for d in 0..HEAD_DIM {
            kh[hd][d] = k[hd * HEAD_DIM + d];
            vh[hd][d] = v[hd * HEAD_DIM + d];
        }
        rmsnorm_head(&mut kh[hd], &a.kn);
        rope(&mut kh[hd], pos);
        ck[hd].push(kh[hd]);
        cv[hd].push(vh[hd]);
    }
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let gqa = HEADS / KV_HEADS;
    let mut ctx = vec![0f32; hq];
    let seq = ck[0].len();
    for hd in 0..HEADS {
        let kv = hd / gqa;
        let mut scores = vec![0f32; seq];
        let mut mx = f32::NEG_INFINITY;
        for t in 0..seq {
            let mut s = 0f32;
            for d in 0..HEAD_DIM {
                s += qh[hd][d] * ck[kv][t][d];
            }
            scores[t] = s * scale;
            mx = mx.max(scores[t]);
        }
        let mut sum = 0f32;
        for t in 0..seq {
            scores[t] = (scores[t] - mx).exp();
            sum += scores[t];
        }
        for d in 0..HEAD_DIM {
            let mut o = 0f32;
            for t in 0..seq {
                o += (scores[t] / sum) * cv[kv][t][d];
            }
            ctx[hd * HEAD_DIM + d] = o;
        }
    }
    gemv_w(&a.o, &ctx, x.len(), hq, 3)
}

#[cfg(feature = "cuda")]
fn init_gpu_attn(layers: &[Layer], hidden: usize) -> anyhow::Result<GpuAttn> {
    use ft_kernel::{device_count, set_device, DevBuffer, Stream};
    let n = device_count()?;
    if n == 0 {
        anyhow::bail!("no device");
    }
    set_device((n as i32) - 1)?;
    let stream = Stream::new()?;
    let max_out = HEADS * HEAD_DIM; // 4096
    let x = DevBuffer::alloc(hidden.max(max_out) * 4)?;
    let y = DevBuffer::alloc(max_out.max(hidden) * 4)?;
    let mut w = Vec::new();
    for L in layers {
        for src in [&L.attn.q, &L.attn.k, &L.attn.v, &L.attn.o] {
            let mut b = DevBuffer::alloc(src.len() * 2)?;
            b.h2d_bytes(src.as_ptr() as *const _, src.len() * 2)?;
            w.push(b);
        }
    }
    Ok(GpuAttn { stream, x, y, w })
}

fn moe_residual(
    h: &mut [f32],
    n: &[f32],
    layer: &Layer,
    hidden: usize,
    e: usize,
    inter: usize,
    k_moe: usize,
    th: usize,
) -> anyhow::Result<()> {
    let (ids, rw) = route(&layer.gate, n, e, hidden, k_moe);
    let ids_i: Vec<i32> = ids.iter().map(|&v| v as i32).collect();
    let mut y = n.to_vec();
    #[cfg(feature = "cpu-simd")]
    {
        ft_kernel::moe_bf16(&mut y, &layer.w13, &layer.w2, &ids_i, &rw, 1, hidden, inter, e, k_moe, th)?;
    }
    #[cfg(not(feature = "cpu-simd"))]
    {
        let _ = (ids_i, th, &layer.w13);
        y.fill(0.0);
    }
    for (a, b) in h.iter_mut().zip(y.iter()) {
        *a += *b;
    }
    Ok(())
}

fn route(gate: &[u16], x: &[f32], e: usize, hidden: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut logits = vec![0f32; e];
    for ei in 0..e {
        let mut s = 0f32;
        for j in 0..hidden {
            s += bf16(gate[ei * hidden + j]) * x[j];
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
    (
        order[..k].iter().map(|&i| i as u32).collect(),
        w.iter().map(|v| v / sum).collect(),
    )
}

fn gemv(w: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows];
    for r in 0..rows {
        let mut s = 0f32;
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            s += bf16(row[c]) * x[c];
        }
        out[r] = s;
    }
    out
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32]) {
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v *= inv * w[i.min(w.len() - 1)];
    }
}

fn rmsnorm_head(x: &mut [f32; HEAD_DIM], w: &[f32]) {
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / HEAD_DIM as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    for d in 0..HEAD_DIM {
        x[d] *= inv * w[d];
    }
}

fn rope(x: &mut [f32; HEAD_DIM], pos: usize) {
    let half = HEAD_DIM / 2;
    for i in 0..half {
        let freq = (pos as f32) / ROPE_THETA.powf(2.0 * i as f32 / HEAD_DIM as f32);
        let (sin, cos) = freq.sin_cos();
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin;
        x[i + half] = b * cos + a * sin;
    }
}

fn bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn load_layers(root: &Path, n: usize, hidden: usize, e: usize, inter: usize) -> anyhow::Result<Vec<Layer>> {
    #[derive(Clone)]
    enum K {
        Aq, Ak, Av, Ao, Aqn, Akn, InN, PostN, Gate, Eg(usize), Eu(usize), Ed(usize),
    }
    let mut by: HashMap<String, Vec<(usize, K, String)>> = HashMap::new();
    let push = |by: &mut HashMap<String, Vec<(usize, K, String)>>, layer: usize, kind: K, name: String| {
        if let Ok(s) = locate_tensor(root, &name) {
            by.entry(s.display().to_string()).or_default().push((layer, kind, name));
        }
    };
    for layer in 0..n {
        push(&mut by, layer, K::InN, format!("model.layers.{layer}.input_layernorm.weight"));
        push(&mut by, layer, K::PostN, format!("model.layers.{layer}.post_attention_layernorm.weight"));
        push(&mut by, layer, K::Aq, format!("model.layers.{layer}.self_attn.q_proj.weight"));
        push(&mut by, layer, K::Ak, format!("model.layers.{layer}.self_attn.k_proj.weight"));
        push(&mut by, layer, K::Av, format!("model.layers.{layer}.self_attn.v_proj.weight"));
        push(&mut by, layer, K::Ao, format!("model.layers.{layer}.self_attn.o_proj.weight"));
        push(&mut by, layer, K::Aqn, format!("model.layers.{layer}.self_attn.q_norm.weight"));
        push(&mut by, layer, K::Akn, format!("model.layers.{layer}.self_attn.k_norm.weight"));
        push(&mut by, layer, K::Gate, format!("model.layers.{layer}.mlp.gate.weight"));
        for ei in 0..e {
            push(&mut by, layer, K::Eg(ei), format!("model.layers.{layer}.mlp.experts.{ei}.gate_proj.weight"));
            push(&mut by, layer, K::Eu(ei), format!("model.layers.{layer}.mlp.experts.{ei}.up_proj.weight"));
            push(&mut by, layer, K::Ed(ei), format!("model.layers.{layer}.mlp.experts.{ei}.down_proj.weight"));
        }
    }
    let empty_attn = || Attn {
        q: vec![0; HEADS * HEAD_DIM * hidden],
        k: vec![0; KV_HEADS * HEAD_DIM * hidden],
        v: vec![0; KV_HEADS * HEAD_DIM * hidden],
        o: vec![0; hidden * HEADS * HEAD_DIM],
        qn: vec![1.0; HEAD_DIM],
        kn: vec![1.0; HEAD_DIM],
        in_norm: vec![1.0; hidden],
        post_norm: vec![1.0; hidden],
    };
    let mut layers: Vec<Layer> = (0..n)
        .map(|_| Layer {
            attn: empty_attn(),
            gate: vec![0; e * hidden],
            w13: vec![0; e * 2 * inter * hidden],
            w2: vec![0; e * hidden * inter],
        })
        .collect();
    let t0 = std::time::Instant::now();
    for (shard, items) in &by {
        let st = SafeTensorFile::open(Path::new(shard))?;
        for (layer, kind, name) in items {
            let tv = st.tensor(name).with_context(|| name.clone())?;
            let bits = tv.as_u16().with_context(|| name.clone())?;
            let L = &mut layers[*layer];
            match kind {
                K::Aq => L.attn.q = bits,
                K::Ak => L.attn.k = bits,
                K::Av => L.attn.v = bits,
                K::Ao => L.attn.o = bits,
                K::Aqn => L.attn.qn = bits.iter().map(|&b| bf16(b)).collect(),
                K::Akn => L.attn.kn = bits.iter().map(|&b| bf16(b)).collect(),
                K::InN => L.attn.in_norm = bits.iter().map(|&b| bf16(b)).collect(),
                K::PostN => L.attn.post_norm = bits.iter().map(|&b| bf16(b)).collect(),
                K::Gate => L.gate = bits,
                K::Eg(ei) => {
                    let off = *ei * 2 * inter * hidden;
                    L.w13[off..off + inter * hidden].copy_from_slice(&bits);
                }
                K::Eu(ei) => {
                    let off = *ei * 2 * inter * hidden + inter * hidden;
                    L.w13[off..off + inter * hidden].copy_from_slice(&bits);
                }
                K::Ed(ei) => {
                    let off = *ei * hidden * inter;
                    L.w2[off..off + hidden * inter].copy_from_slice(&bits);
                }
            }
        }
    }
    eprintln!("  weights loaded in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(layers)
}

struct U16T { bits: Vec<u16>, shape: Vec<usize> }

fn load_u16(root: &Path, name: &str) -> anyhow::Result<U16T> {
    let st = SafeTensorFile::open(&locate_tensor(root, name)?)?;
    let tv = st.tensor(name).ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
    Ok(U16T { bits: tv.as_u16()?, shape: tv.shape.to_vec() })
}

fn load_f32_from_bf16(root: &Path, name: &str) -> anyhow::Result<Vec<f32>> {
    Ok(load_u16(root, name)?.bits.iter().map(|&b| bf16(b)).collect())
}
