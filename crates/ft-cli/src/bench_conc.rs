//! 并发考核：对 control/worker 扫 concurrency，报聚合吞吐和时延分位。
use anyhow::Context;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub async fn run(
    url: String,
    model: String,
    conc_list: Vec<usize>,
    max_tokens: usize,
    prompt: String,
) -> anyhow::Result<()> {
    let base = url.trim_end_matches('/').to_string();
    let client = reqwest::Client::builder().http1_only().tcp_nodelay(true).build()?;
    println!(
        "bench-conc  target={base}  model={model}  max_tokens={max_tokens}  conc={conc_list:?}"
    );
    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
        "N", "agg_tok/s", "p50_ms", "p99_ms", "ok/fail", "tok/req"
    );
    for n in conc_list {
        let r = one_level(&client, &base, &model, n, max_tokens, &prompt).await?;
        println!(
            "{:>6}  {:>10.1}  {:>10.0}  {:>10.0}  {:>4}/{:<3}  {:>8.0}",
            n, r.agg_tps, r.p50_ms, r.p99_ms, r.ok, r.fail, r.tok_per_req
        );
    }
    Ok(())
}

struct Level {
    agg_tps: f64,
    p50_ms: f64,
    p99_ms: f64,
    ok: u32,
    fail: u32,
    tok_per_req: f64,
}

async fn one_level(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    n: usize,
    max_tokens: usize,
    prompt: &str,
) -> anyhow::Result<Level> {
    let done_tok = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let mut joins = Vec::new();
    for i in 0..n {
        let c = client.clone();
        let url = format!("{base}/v1/chat/completions");
        let body = json!({
            "model": model,
            "messages": [{"role":"user","content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        });
        let tok = done_tok.clone();
        joins.push(tokio::spawn(async move {
            let sid = format!("conc-{n}-{i}");
            let t = Instant::now();
            let resp = c
                .post(&url)
                .header("content-type", "application/json")
                .header("x-blend-session", &sid)
                .json(&body)
                .send()
                .await
                .context("send")?;
            let status = resp.status();
            let v: serde_json::Value = resp.json().await.context("json")?;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if !status.is_success() {
                anyhow::bail!("http {status}: {v}");
            }
            let ct = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
            tok.fetch_add(ct, Ordering::Relaxed);
            Ok::<(f64, u64), anyhow::Error>((ms, ct))
        }));
    }
    let mut lats = Vec::new();
    let mut toks = Vec::new();
    let mut fail = 0u32;
    for j in joins {
        match j.await {
            Ok(Ok((ms, ct))) => {
                lats.push(ms);
                toks.push(ct);
            }
            _ => fail += 1,
        }
    }
    let wall = t0.elapsed().as_secs_f64().max(1e-6);
    let total_tok = done_tok.load(Ordering::Relaxed) as f64;
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| -> f64 {
        if lats.is_empty() {
            return 0.0;
        }
        let i = ((lats.len() as f64 - 1.0) * q).round() as usize;
        lats[i.min(lats.len() - 1)]
    };
    Ok(Level {
        agg_tps: total_tok / wall,
        p50_ms: p(0.50),
        p99_ms: p(0.99),
        ok: lats.len() as u32,
        fail,
        tok_per_req: if toks.is_empty() {
            0.0
        } else {
            toks.iter().sum::<u64>() as f64 / toks.len() as f64
        },
    })
}

pub fn parse_list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|x| x.trim().parse().ok())
        .filter(|&n| n > 0)
        .collect()
}

#[allow(dead_code)]
fn _unused_duration() -> Duration {
    Duration::from_secs(0)
}
