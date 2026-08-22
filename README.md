# blend

自研 MoE 推理服务。技术路线：fork [FreeToken](https://github.com/FlashML-org/FreeToken)，
按 [docs/freetoken-rust-architecture.md](docs/freetoken-rust-architecture.md) 的设计逐步 Rust 化。

**核心策略**：Rust 重写控制面（服务/调度/内存）与 CPU 计算面，GPU 内核（CUDA/Triton）
通过 FFI 复用，FFI + 数值对拍缝合。

## 当前状态（P0 阶段）

- [x] cargo workspace：13 crates 骨架，依赖分层（ft-core 零依赖、ft-engine 无 async）
- [x] `ft-core` — 核心类型（Dtype/ModelConfig/ChatRequest/ReqState）
- [x] `ft-memory` — generation-based SlotTable（防 use-after-free）、页式 KV pool trait
- [x] `ft-moe` — q\* 带宽自适应策略（纯函数，含 PRO 6000 实测画像）、MoePlan、CPU MoE 参考执行器
- [x] `ft-engine` — 连续批处理调度状态机（chunked prefill / decode 优先 / 抢占）
- [x] `ft-kernel-sys` / `ft-kernel` — CUDA FFI 绑定层（feature = "cuda"，当前 stub）
- [x] `ft-server` — OpenAI 兼容 API（/v1/models、/v1/chat/completions、SSE 流式）
- [x] `ft-cli` — `blend serve` / `blend bench-bw`

引擎当前为 stub（回显 prompt），协议已定型；P2 替换为真实 GPU 引擎时 server 侧零改动。

## 快速开始

```bash
cargo build
cargo test

# 启动服务
./target/debug/blend serve --port 1919

# 冒烟
curl -s localhost:1919/v1/models
curl -s -X POST localhost:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"blend-stub","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'

# 标定 q* 画像（当前接受外部测量值，P2 接入真实微基准）
./target/debug/blend bench-bw --cpu-gbps 124.4 --pcie-gbps 57.7
```

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | fork + 控制面（API/计量/路由/动态调参） | 🚧 骨架已就绪 |
| P1 | Rust server + tokenizer + scheduler 前端 | 🚧 本仓库 |
| P2 | Rust engine core + 内存管理，GPU kernel FFI | 待开始 |
| P3 | CPU MoE AVX512 内核、FTW v2 加载、内核逐步替换 | 待开始 |

## 文档

- [FreeToken 在 RTX PRO 6000 8 卡机上的部署与测试报告](docs/freetoken-6000pro-test-report.md)
- [FreeToken Rust 化重构架构设计](docs/freetoken-rust-architecture.md)
