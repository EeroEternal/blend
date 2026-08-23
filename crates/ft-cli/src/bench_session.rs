//! 同 session 多轮：测 TTFT 是否因 worker 前缀缓存下降。
use anyhow::Context;
use serde_json::{json, Value};
use std::time::Instant;

pub async fn run(
    url: String,
    model: String,
    turns: usize,
    max_tokens: usize,
    sticky: bool,
) -> anyhow::Result<()> {
    let base = url.trim_end_matches('/');
    let client = reqwest::Client::builder().http1_only().tcp_nodelay(true).build()?;
    println!(
        "bench-session  url={base}  turns={turns}  sticky={sticky}  max_tokens={max_tokens}"
    );
    println!("{:>4}  {:>10}  {:>10}  {:>6}  {:>8}  worker", "turn", "ttft_ms", "total_ms", "tok", "sid");

    let mut history = vec![json!({"role":"user","content":"Explain mixture-of-experts in one sentence, then wait."})];
    for t in 0..turns {
        let sid = if sticky {
            "agent-sess-v23".to_string()
        } else {
            format!("agent-sess-v23-turn-{t}")
        };
        let body = json!({
            "model": model,
            "messages": history,
            "max_tokens": max_tokens,
            "stream": true,
            "temperature": 0.0,
        });
        let t0 = Instant::now();
        let mut resp = client
            .post(format!("{base}/v1/chat/completions"))
            .header("content-type", "application/json")
            .header("x-blend-session", &sid)
            .json(&body)
            .send()
            .await
            .context("send")?;
        let worker = resp
            .headers()
            .get("x-blend-worker")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("-")
            .to_string();
        let mut ttft = None;
        let mut acc = String::new();
        let mut tok = 0u32;
        while let Some(chunk) = resp.chunk().await.context("chunk")? {
            if ttft.is_none() {
                ttft = Some(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let s = String::from_utf8_lossy(&chunk);
            for line in s.lines() {
                let line = line.trim();
                let Some(rest) = line.strip_prefix("data:") else { continue };
                let rest = rest.trim();
                if rest == "[DONE]" || rest.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(rest) {
                    if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                        acc.push_str(c);
                        tok += 1;
                    }
                }
            }
        }
        let total = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{:>4}  {:>10.0}  {:>10.0}  {:>6}  {:>8}  {worker}",
            t + 1,
            ttft.unwrap_or(total),
            total,
            tok,
            if sticky { "same" } else { "new" }
        );
        history.push(json!({"role":"assistant","content": acc}));
        history.push(json!({"role":"user","content": format!("Continue with point number {}.", t + 2)}));
    }
    Ok(())
}
