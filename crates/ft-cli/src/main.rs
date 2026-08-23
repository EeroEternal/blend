//! blend CLI：serve / bench-bw。

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "blend", version, about = "blend — 自研 MoE 推理服务（fork 自 FreeToken 路线）")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 启动 OpenAI 兼容 API 服务（当前为 stub 引擎，P2 接 GPU 内核）
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 1919)]
        port: u16,
        #[arg(long, default_value = "blend-stub")]
        model: String,
    },
    /// 标定带宽画像并保存 q\* 策略参数
    BenchBw {
        #[arg(long, default_value_t = 124.4)]
        cpu_gbps: f64,
        #[arg(long, default_value_t = 57.7)]
        pcie_gbps: f64,
        #[arg(long, default_value = "~/.cache/blend/benchbw.json")]
        out: String,
    },
    /// CPU MoE 执行器数值对拍：读 fixtures 目录（manifest.json + f32 bins）
    Parity {
        /// fixture 目录：manifest.json / h_in.f32 / w13.f32 / w2.f32 / h_golden.f32
        #[arg(long)]
        dir: String,
        #[arg(long, default_value_t = 1e-3)]
        tol: f32,
    },
    /// 多线程内存读带宽（STREAM-like），用于标定 q\* 画像的 cpu_gbps
    MemBench {
        /// 缓冲区大小 GiB
        #[arg(long, default_value_t = 8)]
        gib: usize,
        #[arg(long, default_value_t = 3)]
        iters: usize,
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
    /// GPU 链路冒烟：设备查询 + H2D + 内核 + D2H + 校验（需 --features cuda 构建）
    GpuSmoke {
        /// 元素个数
        #[arg(long, default_value_t = 1024)]
        n: usize,
    },
    /// CPU MoE 执行器吞叶基准（DSV4 形状默认值）
    MoeBench {
        #[arg(long, default_value_t = 8)]
        tokens: usize,
        #[arg(long, default_value_t = 4096)]
        hidden: usize,
        #[arg(long, default_value_t = 2048)]
        inter: usize,
        #[arg(long, default_value_t = 256)]
        experts: usize,
        #[arg(long, default_value_t = 6)]
        k: usize,
        #[arg(long, default_value_t = 5)]
        iters: usize,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();
    match Cli::parse().cmd {
        Cmd::Serve { host, port, model } => {
            let engine = ft_server::spawn_engine(model);
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(async move {
                let app = ft_server::api::router(engine);
                let addr = format!("{host}:{port}");
                let listener = tokio::net::TcpListener::bind(&addr).await?;
                tracing::info!("API server is ready to serve on http://{addr}");
                axum::serve(listener, app).await?;
                anyhow::Ok(())
            })?;
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
        Cmd::Parity { dir, tol } => {
            parity::run(std::path::Path::new(&dir), tol)?;
        }
        Cmd::MoeBench { tokens, hidden, inter, experts, k, iters } => {
            moe_bench::run(tokens, hidden, inter, experts, k, iters);
        }
        Cmd::MemBench { gib, iters, threads } => {
            mem_bench::run(gib, iters, threads);
        }
        Cmd::GpuSmoke { n } => {
            gpu_smoke::run(n)?;
        }
    }
    Ok(())
}

mod gpu_smoke;
mod mem_bench;
mod moe_bench;
mod parity;

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return std::path::PathBuf::from(home).join(rest).display().to_string();
        }
    }
    p.to_string()
}
