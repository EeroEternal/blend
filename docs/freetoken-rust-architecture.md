# FreeToken Rust 化重构架构设计

> 日期：2026-08-22
> 前置文档：[FreeToken 在 RTX PRO 6000 8 卡机上的部署与测试报告](./freetoken-6000pro-test-report.md)
> 目标：fork [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)，构建自有、可控的推理服务，工程语言迁移到 Rust

---

## 0. 总体判断：哪些能 Rust 化，哪些不能

FreeToken 现状（v0.1.2）：约 71K 行 Python + CUDA csrc + 25 个 Triton kernel 文件。资产分布：

| 层 | 内容 | Rust 重写可行性 |
|---|---|---|
| **GPU 内核** | Triton kernel（nvfp4/fp8 MoE、DSA sparse attention、RoPE...）+ csrc（cpu_moe AVX512、jit）+ flashinfer/sglang-kernel 依赖 | ❌ **不重写**。Rust GPU 生态（candle/burn）没有 NVFP4、DSA、hybrid-attention 等前沿内核，重写等于重做两年研究 |
| **CPU MoE 执行器** | `moe/cpu_executor.py`，AVX512BF16 就地计算（实测 ~155 GB/s） | ✅ 最适合 Rust 化：性能敏感 + 纯计算，rayon + `std::arch` 可持平或超越 |
| **引擎核心** | scheduler / CUDA graph / KV pool / expert banks / q\* 策略 / 内存预算 | ✅ 收益最大：内存安全、无 GIL、延迟可控 |
| **服务面** | API server / tokenizer / daemon / TUI | ✅ 最成熟：tokio + axum 直接替代 |

**核心策略：Rust 重写"控制面 + 内存面 + CPU 计算面"，GPU 内核通过 FFI 复用。**
FreeToken 自带 `kernel/aot.py` 将 Triton AOT 编译为 cubin，FFI 路径是现成的。

---

## 1. 总体架构：三层分离

```
┌─────────────────────────────────────────────────┐
│  ft-server (tokio async)                        │
│  OpenAI/Anthropic API · SSE · 计量 · 多模型路由   │
└──────────────────┬──────────────────────────────┘
                   │ flume channel (Req/Resp 流)
┌──────────────────▼──────────────────────────────┐
│  ft-engine (专用 OS 线程, 单线程事件循环)          │
│  scheduler · continuous batching · CUDA graph    │
│  KV pool · radix cache · elastic memory budget   │
├─────────────────────────────────────────────────┤
│  ft-moe / ft-attention / ft-models (trait 分发)  │
│  ├─ CPU executor (rayon + AVX512, 纯 Rust)      │
│  └─ GPU kernels ──FFI──> libftkernels.so        │
│       (FreeToken csrc + AOT cubin + flashinfer)  │
└─────────────────────────────────────────────────┘
```

### 进程模型关键决策

FreeToken 用多进程（tokenizer worker + scheduler + TP workers，见 `utils/mp.py`）。Rust 版改为：

- 单实例 = **1 个 tokio runtime（server）+ 1 个 engine 线程 + N 个 rayon worker（CPU MoE）**
- engine 必须是普通 OS 线程而非 async 任务——CUDA Graph 捕获、pinned memory 注册、stream 语义都要求确定性执行；**不要把 async 污染进 decode loop**
- server ↔ engine 用 `flume` 有界 channel 传消息（请求 / 流式 token），无共享锁

---

## 2. Cargo Workspace 布局

```
freetoken-rs/
├── crates/
│   ├── ft-core/         # 零依赖核心类型: Request, Seq, Batch, Config, Dtype, DevTensor view
│   ├── ft-tokenizer/    # tokenizers crate 封装 + 增量 detokenize 流
│   ├── ft-loader/       # safetensors mmap / FTW 快速加载 / expert bank 打包
│   ├── ft-memory/       # pinned allocator · 页式 KV pool · radix cache · LRU slot 表
│   ├── ft-kernel-sys/   # -sys crate: libftkernels.so 的 bindgen 绑定
│   ├── ft-kernel/       # 安全封装: attention/moe/quant/sampling 调用
│   ├── ft-moe/          # offload cache · q* 策略(纯函数) · Rust CPU executor
│   ├── ft-attention/    # fa/dsa/dsv4_sparse backend trait + 分发
│   ├── ft-models/       # Model trait + glm/dsv4/qwen 具体实现
│   ├── ft-engine/       # scheduler 状态机 · CUDA graph · decode/prefill loop
│   ├── ft-server/       # axum API 层 (OpenAI/Anthropic/SSE)
│   ├── ft-bench/        # benchbw 标定（q* 硬件画像）
│   └── ft-cli/          # clap: serve/shell/bench/checkpoint
└── kernels/             # 从 FreeToken fork 的 csrc + AOT 构建脚本 → libftkernels.so
```

依赖原则：
- `ft-core` 不依赖任何 runtime
- `ft-engine` 不依赖 tokio
- 只有 `ft-server` 和 `ft-cli` 接触 async

---

## 3. 核心 trait 设计（对照 FreeToken 的 ABC 类）

```rust
// ft-moe —— 对应 moe/base.py::BaseMoeBackend
pub trait MoeBackend: Send + Sync {
    /// 返回本步执行计划，让 engine 可观测/可干预（可控性关键点）
    fn plan(&self, routing: &RoutingInfo) -> MoePlan;   // Fetch(n) + CpuCompute(m)
    fn forward(&self, h: &mut DevTensor, plan: &MoePlan, stream: &Stream);
}

// ft-memory —— 对应 kvcache/ 的各 pool
pub trait KvPool: Send {
    fn alloc(&mut self, req: &ReqState) -> Result<SeqHandle, Oom>;
    fn free(&mut self, h: SeqHandle);
    fn match_prefix(&self, tokens: &[u32]) -> (SeqHandle, usize); // radix 匹配
}

// ft-models —— 对应 models/ 各架构目录
pub trait Model: Send {
    fn prefill(&mut self, batch: &PrefillBatch, s: &Stream) -> Result<()>;
    fn decode(&mut self, batch: &DecodeBatch, s: &Stream) -> Result<Logits>;
    /// 弹性内存：运行时改预算（对应 FreeToken 的 elastic resize）
    fn resize_budget(&mut self, moe_cache: usize, kv_pages: usize) -> Result<()>;
}
```

设计要点：

1. **q\* 策略做成纯函数**，从 profile 文件（等价于 `~/.cache/freetoken/benchbw.json`）读取硬件画像参数。这是 fork 后最常调的东西，单独放 `ft-moe::policy` 模块，配单元测试和 criterion 基准。
2. `MoePlan` 显式化执行计划（多少 miss 走 PCIe、多少 CPU 就地算），engine 层可以记录、限流、甚至覆盖——这就是"更可控"的落点。
3. 所有 backend trait 要求 `Send + Sync` 但内部通过 stream 句柄操作 GPU，避免 `&mut Tensor` 跨线程问题。

---

## 4. Rust 化收益最大的三个子系统

### 4.1 内存管理（`ft-memory`）

- **Pinned allocator**：`cudaHostAlloc` 包成 `Allocator` trait；专家 bank 是几块 100GB+ 的 pinned 大缓冲 + `slotmap` 管理 slot。"expert 正在被 CPU 线程计算时不会被换出"这类约束用 Rust 生命周期变成编译期保证。
- **权重加载**：`memmap2` mmap safetensors → 零拷贝 `cudaMemcpyAsync` 直达最终布局。FTW 格式（按最终布局预打包）的加载就是顺序 mmap——当前 GLM-5.2 专家池加载需 3.4 分钟（实测 2GB/s），mmap + 批量 DMA 有数量级优化空间。
- **KV pool**：页表 = `Vec<PageId>` + generation counter 防 use-after-free。

### 4.2 CPU MoE 执行器（`ft-moe::cpu`）

- `rayon` 按 (layer, expert) 二维并行
- `std::arch::x86_64` 手写 AVX512-BF16 dot + VNNI 反量化路径（对齐 FreeToken csrc 的 `cpu_moe` 实现）
- 运行时 CPU ISA 检测（对应 CLI 的 `--isa`），fallback 到 `gemm` crate
- 纯批量计算、无状态，最容易做 golden test 数值对拍

### 4.3 调度器（`ft-engine`）

```rust
enum Step {
    Prefill(Chunk),
    Decode(Batch),
    Preempt(ReqId),
    Resize(Budget),     // 弹性内存调整
}
```

- 显式状态机 + 单线程循环
- 每步 deadline 感知：decode 步间插入 prefill chunk，TTFT/TPOT 权衡可配置——fork 后差异化体验的核心旋钮
- CUDA graph：cudarc 的 stream/graph API 捕获 bs ∈ {1,2,4,8,...}，graph 内 kernel 参数走 `ft-kernel-sys` 原始绑定

---

## 5. Kernel FFI 具体做法

```rust
// ft-kernel-sys: 手写/bindgen 绑定
extern "C" {
    fn ft_nvfp4_fused_moe(/* raw pointers */ ...) -> c_int;
    fn ft_dsa_sparse_attention(/* ... */) -> c_int;
}
```

1. `kernels/` 目录直接 fork FreeToken 的 `kernel/csrc` + CMake，产出 `libftkernels.so`（CUDA 13 工具链编译）
2. Triton 内核用 FreeToken 现成的 AOT 机制（`kernel/aot.py`）预编译成 cubin 集合，Rust 侧 `cuModuleLoad` + `cuLaunchKernel`（cudarc 提供 module API）
3. flashinfer / sglang-kernel 的稳定导出符号直接 `dlopen`
4. **数值对拍是生命线**：写一个 `ft-parity` 工具，同一输入分别喂 Python 引擎与 Rust 引擎，逐层断言 max-abs-diff < 1e-2（fp4 量化容差）

---

## 6. 分阶段迁移路线

不要全量重写，按可控性收益排序：

| 阶段 | 内容 | 产出 | 周期感 |
|---|---|---|---|
| **P0** | fork + Rust control-plane：API 网关、计量计费、多实例路由、动态调 q\*/cache-size（经 FreeToken `ft ctl` HTTP 接口） | 立刻获得"可控"，engine 不动 | 周级 |
| **P1** | Rust server + tokenizer + scheduler 前端；Python engine 降级为 IPC worker | 服务面全 Rust | 月级 |
| **P2** | Rust engine core：内存管理 + expert banks + KV pool + 调度循环；GPU kernel 走 FFI | 摆脱 Python GIL/进程模型 | 季度级 |
| **P3** | CPU MoE executor 纯 Rust、FTW v2 加载格式、按需替换更多内核 | 完全自主 | 持续 |

P0 即可解决约 80% 的"可控"诉求（FreeToken 本身暴露 `ft ctl` 与 HTTP 管理面）；P2 是重头戏；P3 持续演进。

---

## 7. 依赖选型

| 用途 | crate |
|---|---|
| CUDA 交互 | `cudarc`（nvrtc / driver / graph API） |
| 异步 / 服务 | `tokio` + `axum` + `tower` |
| engine 通道 | `flume` |
| CPU 并行 | `rayon` |
| tokenizer | `tokenizers`（HF 官方 Rust 库，FreeToken 本就依赖） |
| 权重 | `memmap2` + `safetensors` |
| 半精度 | `half`（bf16/f16）+ `bytemuck` |
| 可观测 | `tracing` + `metrics` + prometheus exporter |
| 错误处理 | `thiserror`（库）/ `eyre`（应用） |
| CLI | `clap` |
| 数据结构 | `slotmap`（expert/KV slot） |
| 基准测试 | `criterion` |

---

## 8. 测试与验收策略

1. **数值对拍（parity）**：每个 kernel wrapper 一个 golden test，输入从 Python 引擎 dump，逐层比较
2. **调度器**：纯逻辑状态机，`cargo test` 全覆盖；模糊测试请求序列防死锁/泄漏
3. **内存**：KV/expert slot 的 generation counter + debug assert；Miri 跑 `ft-memory` 的 unsafe 边界
4. **性能回归**：criterion 基准锁定 CPU MoE 吞吐（目标 ≥155 GB/s 等效带宽）；e2e 以本报告实测数据为基线（DSV4-Flash ≈32 tok/s、GLM-5.2 ≈16 tok/s @ PRO 6000）
5. **CI**：GPU job 用单卡 runner 跑 parity + e2e smoke；纯逻辑 job 无 GPU 依赖

---

## 9. 一句话总结

**把 Rust 用在它赢的地方（内存安全的状态机、CPU 带宽压榨、服务面），把 CUDA/Triton 留在它赢的地方（GPU 内核），用 FFI + 数值对拍缝合。**

fork 后真正的差异化价值在控制面能力：q\* 策略可编程、弹性内存可调度、多模型路由可治理。
