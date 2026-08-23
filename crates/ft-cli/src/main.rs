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
    /// 加载一整层 Qwen3-MoE（router + 全部专家），真实 top-k 前向
    MoeLayer {
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 1)]
        tokens: usize,
    },
    /// 从 HF 目录加载真实专家权重并 CPU/GPU 对拍
    RealExpert {
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 0)]
        expert: usize,
    },
    /// 单专家 GPU SwiGLU vs CPU naive 对拍（需 --features cuda）
    GpuFfnParity {
        #[arg(long, default_value_t = 256)]
        hidden: usize,
        #[arg(long, default_value_t = 128)]
        inter: usize,
        #[arg(long, default_value_t = 20)]
        iters: usize,
    },
    /// 实测 PCIe H2D 带宽（需 --features cuda）
    PcieBench {
        #[arg(long, default_value_t = 256)]
        mib: usize,
        #[arg(long, default_value_t = 5)]
        iters: usize,
    },
    /// Hybrid 路径冒烟：LRU 缓存 + q* 拆分 + 真实 SIMD 内核
    HybridSmoke {
        #[arg(long, default_value_t = 8)]
        steps: usize,
        #[arg(long, default_value_t = 43)]
        layers: usize,
        /// GPU 专家缓存槽数（FreeToken DSV4 自动分配 5835）
        #[arg(long, default_value_t = 5835)]
        cache_slots: usize,
        #[arg(long, default_value_t = 0)]
        threads: usize,
        /// 真 PCIe H2D（需 --features cuda）；Fetch 走 cudaMemcpyAsync
        #[arg(long, default_value_t = false)]
        real_pcie: bool,
    },
    /// 调度器 + SIMD MoE 端到端冒烟（走 admit/prefill/decode 循环）
    DecodeSmoke {
        /// decode 步数
        #[arg(long, default_value_t = 8)]
        steps: usize,
        /// 每步模拟的 MoE 层数（DSV4=43 / GLM-5.2=75）
        #[arg(long, default_value_t = 4)]
        layers: usize,
        /// 使用 DSV4 真实形状（H4096 I2048 E256 k6）；默认小形状便于冒烟
        #[arg(long, default_value_t = false)]
        full: bool,
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
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
        /// naive | simd（BF16 口径建议 tol ≥ 1e-2）
        #[arg(long, default_value = "naive")]
        kernel: String,
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
        /// naive | simd（simd 需 --features cpu-simd 构建）
        #[arg(long, default_value = "naive")]
        kernel: String,
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
        /// 工作线程数（0=物理核数）
        #[arg(long, default_value_t = 0)]
        threads: usize,
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
        Cmd::MoeLayer { model, layer, tokens } => {
            moe_layer::run(&model, layer, tokens)?;
        }
        Cmd::RealExpert { model, layer, expert } => {
            real_expert::run(&model, layer, expert)?;
        }
        Cmd::GpuFfnParity { hidden, inter, iters } => {
            gpu_ffn_parity::run(hidden, inter, iters)?;
        }
        Cmd::PcieBench { mib, iters } => {
            pcie_bench::run(mib, iters)?;
        }
        Cmd::HybridSmoke { steps, layers, cache_slots, threads, real_pcie } => {
            hybrid_smoke::run(steps, layers, cache_slots, threads, real_pcie);
        }
        Cmd::DecodeSmoke { steps, layers, full, threads } => {
            decode_smoke::run(steps, layers, full, threads);
        }
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
        Cmd::Parity { dir, tol, kernel } => {
            parity::run(std::path::Path::new(&dir), tol, &kernel)?;
        }
        Cmd::MoeBench { kernel, tokens, hidden, inter, experts, k, iters, threads } => {
            moe_bench::run(&kernel, tokens, hidden, inter, experts, k, iters, threads);
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

mod decode_smoke;
mod gpu_ffn_parity;
mod gpu_smoke;
mod hybrid_smoke;
mod pcie_bench;
mod moe_layer;
mod real_expert;
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
