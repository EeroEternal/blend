//! blend：单机推理调度（生产路径 = control → FreeToken / PyTorch worker）

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "blend",
    version,
    about = "blend-control：多 worker 调度 / 会话粘滞 / 并发考核。算力层是 ft serve（FreeToken）。"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 反代到 FreeToken / torch worker
    Control {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// 可重复。例: --worker http://127.0.0.1:1930 --worker 'http://127.0.0.1:1940#tp=2'
        #[arg(long = "worker", required = true)]
        workers: Vec<String>,
    },
    /// 拉起一个 ft serve（不阻塞）
    Spawn {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0")]
        gpus: String,
        #[arg(long, default_value_t = 1)]
        tp: usize,
        #[arg(long, default_value_t = 1940)]
        port: u16,
        #[arg(long, default_value = "ft")]
        bin: String,
    },
    /// 同 session 多轮 TTFT
    BenchSession {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        #[arg(long, default_value = "Qwen3-30B-A3B-Instruct")]
        model: String,
        #[arg(long, default_value_t = 6)]
        turns: usize,
        #[arg(long, default_value_t = 48)]
        max_tokens: usize,
        #[arg(long, default_value_t = false)]
        fresh: bool,
    },
    /// 并发扫 N
    BenchConc {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        #[arg(long, default_value = "Qwen3-30B-A3B-Instruct")]
        model: String,
        #[arg(long, default_value = "1,2,4,8")]
        concurrency: String,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        #[arg(long, default_value = "Write a short paragraph about mixture of experts.")]
        prompt: String,
    },
    /// 标定 q* 画像（放置用）
    BenchBw {
        #[arg(long, default_value_t = 124.4)]
        cpu_gbps: f64,
        #[arg(long, default_value_t = 57.7)]
        pcie_gbps: f64,
        #[arg(long, default_value = "~/.cache/blend/benchbw.json")]
        out: String,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    match Cli::parse().cmd {
        Cmd::Control { host, port, workers } => {
            tracing::info!(?workers, "starting blend-control");
            let gw = ft_server::Gateway::new(workers)?;
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            rt.block_on(async move {
                let addr = format!("{host}:{port}");
                let listener = tokio::net::TcpListener::bind(&addr).await?;
                tracing::info!("blend-control ready on http://{addr}");
                axum::serve(listener, gw.router()).await?;
                anyhow::Ok(())
            })?;
        }
        Cmd::Spawn { model, gpus, tp, port, bin } => spawn_ft(&bin, &model, &gpus, tp, port)?,
        Cmd::BenchSession { url, model, turns, max_tokens, fresh } => {
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            rt.block_on(bench_session::run(url, model, turns, max_tokens, !fresh))?;
        }
        Cmd::BenchConc { url, model, concurrency, max_tokens, prompt } => {
            let list = bench_conc::parse_list(&concurrency);
            if list.is_empty() {
                anyhow::bail!("--concurrency 为空");
            }
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            rt.block_on(bench_conc::run(url, model, list, max_tokens, prompt))?;
        }
        Cmd::BenchBw { cpu_gbps, pcie_gbps, out } => {
            let path = expand_tilde(&out);
            let pol = ft_bench::save_profile(std::path::Path::new(&path), cpu_gbps, pcie_gbps)?;
            println!(
                "backend={:?} fetch_fraction={:.1}% → {}",
                pol.backend(),
                pol.fetch_fraction() * 100.0,
                path
            );
        }
    }
    Ok(())
}

mod bench_conc;
mod bench_session;

fn spawn_ft(bin: &str, model: &str, gpus: &str, tp: usize, port: u16) -> anyhow::Result<()> {
    use std::process::{Command, Stdio};
    let log = format!("/tmp/blend-worker-{port}.log");
    let file = std::fs::File::create(&log)?;
    let mut cmd = Command::new(bin);
    cmd.arg("serve")
        .arg("--model")
        .arg(model)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env("CUDA_VISIBLE_DEVICES", gpus)
        .stdout(Stdio::from(file.try_clone()?))
        .stderr(Stdio::from(file));
    if tp > 1 {
        cmd.arg("--tp-size").arg(tp.to_string());
    }
    let child = cmd.spawn()?;
    println!("spawned pid={}  gpus={gpus} tp={tp} port={port}  log={log}", child.id());
    println!(
        "register: --worker http://127.0.0.1:{port}{}",
        if tp > 1 { format!("#tp={tp}") } else { String::new() }
    );
    std::mem::forget(child);
    Ok(())
}

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return std::path::PathBuf::from(home).join(rest).display().to_string();
        }
    }
    p.to_string()
}
