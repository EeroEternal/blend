# blend 工作状态与待办（P2 进行中）

> 更新：2026-08-23
> 本文档是滚动更新的工作台账：记录已完成的工作、正在进行的分析结论、以及下一步的具体任务。

---

## 一、总体进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 骨架 | ✅ 完成 | 13 crates，27 单测，双平台绿（commit `36722fb`） |
| P1 真机验证 | ✅ 完成 | parity 2.7e-12 / moe-bench 基线 8.7 GB/s / mem-bench 标定（`eeac06c`, `fac9cf3`） |
| **P2 引擎核心** | 🚧 **进行中** | GPU FFI 链路已通；CPU MoE 内核吸收中 |
| P3 内核自主化 | ⬜ 未开始 | SIMD/内核逐步替换 |

相关文档：
- 架构设计：[freetoken-rust-architecture.md](./freetoken-rust-architecture.md)
- P1 验证记录：[blend-p1-verification.md](./blend-p1-verification.md)
- SMT 坑记录：[pitfall-smt-bandwidth.md](./pitfall-smt-bandwidth.md)
- FreeToken 部署报告：[freetoken-6000pro-test-report.md](./freetoken-6000pro-test-report.md)

## 二、已完成的关键资产

### 代码（crates/）
| 模块 | 内容 | 验证方式 |
|---|---|---|
| `ft-core` | Dtype(含 NVFP4/DS_FP4)/ModelConfig/ChatRequest | 编译期+单测 |
| `ft-memory` | SlotTable(generation 防 stale handle)、页式 KV pool trait | stale handle 复现测试 |
| `ft-moe::policy` | q\* 策略纯函数 + PRO6000 实测画像 | fetch_fraction≈18.7% 对齐实测 |
| `ft-moe::cpu` | NaiveF32Executor（golden 参考实现） | torch 对拍 diff=2.7e-12 |
| `ft-engine` | 连续批处理调度状态机（chunked prefill/decode优先/抢占） | 5 个流程单测 |
| `ft-kernel(-sys)` | CUDA runtime 绑定 + DevBuffer RAII + vector_add（feature=cuda） | 真机 gpu-smoke PASS |
| `ft-server` | OpenAI 兼容 API + SSE + engine 线程桥 | Linux 冒烟 PASS |
| `ft-cli` | serve/bench-bw/parity/moe-bench/mem-bench/gpu-smoke | 全部真机运行过 |

### 真机环境（bodesi@39.183.171.3:2208）
- `~/blend` — 源码 + fixtures + release 二进制（含 cuda feature）
- `~/blend/kernels/build/libftkernels.so` — sm_100，已链接通过
- `~/freetoken-venv` — torch 环境（生成 golden fixture 用）
- CUDA 13.0 工具链 `/usr/local/cuda-13.0`；GPU 4/5 空闲

### 关键基线数据（EPYC 9355）
| 指标 | 数值 | 来源 |
|---|---|---|
| CPU STREAM 读 | 124.4 GB/s (FreeToken) / 225.7 GB/s (64t 上界) | benchbw / mem-bench |
| PCIe H2D | 57.7 GB/s | benchbw |
| naive f32 CPU MoE | 8.7 GB/s @ DSV4 形状 | moe-bench |
| FreeToken AVX512BF16 | 155 GB/s 同形状 | benchbw → **18× 优化空间** |
| q\* fetch 比例 | ~18.4–18.7% | hybrid 后端标定 |

## 三、正在分析：FreeToken cpu_moe_ext.cpp 抽取方案

源文件：`python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`（2150 行，torch extension）

### 结构分析结论（已完成的侦察）
```
行 54–230    纯函数层: bf16↔f32 转换、dot_scalar/f16/avx512f/avx512bf16/avx512vnni、
             nvfp4/e4m3 解码 + dot_nvfp4_{scalar,avx512,avx2,i8_vnni}   ← 【可独立编译】
行 572–660   cuMemOp64 dlopen 桥(GPU-CPU 同步原语)                      ← 可选,依赖 driver API
行 1076–1230 MoeTask 数据结构(banks/ids/weights 布局)                    ← 【需要适配】
行 1344–2100 CpuMoeExecutor 类: 工作线程池+core pinning+barrier          ← 【依赖 pthread/sched,
                                                                            无 torch 也可编,但接口是 C++ 类】
PYBIND11 部分(尾部)                                                      ← 丢弃,Rust 直接 extern "C"
```

关键发现：**计算核心不依赖 torch**（只有 extension 胶水层用），`immintrin.h` + `pthread`
即可独立编译。torch 依赖集中在 PYBIND11_MODULE 和 Tensor 参数转换处。

### 抽取策略
1. 不改 FreeToken 源文件——在 `kernels/cpu_moe_shim.cpp` 写一个薄壳：
   - `#include` 原 cpp 中纯计算段落无法直接 include（单文件混合），改为**拷贝计算函数段**
     到 shim（标注来源 commit），或者用 `-fPIC` 编译原文件 + 链接 torch stub —— 前者更干净
2. shim 暴露 C ABI：
   ```c
   // bf16 专家 bank 的 MoE 前向（语义对齐 ft_moe::NaiveF32Executor）
   int ft_cpu_moe_bf16(
       float* h,            // [T,H] in/out
       const uint16_t* w13, // [E,2I,H] bf16 banks
       const uint16_t* w2,  // [E,H,I] bf16 banks
       const int32_t* ids,  // [T,K]
       const float* rw,     // [T,K]
       int T, int H, int I, int E, int K,
       int threads, int isa);  // isa: 0=scalar 3=avx512bf16
   ```
3. Rust 侧 `ft-kernel-sys` 加声明 → `ft-cli parity --backend avx512` 与 NaiveF32Executor 对拍

## 四、下一步任务清单（按序执行）

### ① AVX512BF16 内核吸收 + 对拍 ✅ 完成（2026-08-23）
- [x] 抽取 dot 函数族到 `kernels/basic/cpu_moe_shim.cpp`（标注来源/Apache-2.0）
- [x] build.sh 编出 `libftcpu.so`（g++ 纯 CPU，无 CUDA 依赖；ISA 运行时探测）
- [x] `ft-kernel-sys` cpu-simd feature + `ft_cpu_moe_bf16`；`ft-kernel::moe_bf16` 安全封装
- [x] 三方对拍全过：
  - case1(小): simd rel=5.9e-3 ✅ / naive rel≈1e-6 ✅
  - case2(H512/I256): simd rel=4.3e-3 ✅ / naive rel=7.4e-7 ✅
- [x] DSV4 真实形状（8tok H4096 I2048 E256 k6，64 物理线程）：
  - **naive f32: 555.7 ms/step = 8.7 GB/s**
  - **simd avx512bf16: 21.2 ms/step = 113.7 GB/s（26×）**
  - 达到 FreeToken 自带池（155.4 GB/s）的 73%
- 过程中修复的 bug：
  - 激活融合下标错误——gate 在块内 `[i]`、up 在 `[I+i]`，不是相邻配对（真机对拍抓出）
  - shim 漏 `immintrin.h`
  - moe-bench 带宽按实际权重宽度计（simd=BF16 2B）

### ①' 持久线程池 ✅ 完成（2026-08-23）
- [x] `WorkerPool`：进程级单例，condvar 唤醒 + 代际 ack；主线程为 0 号参与者
- [x] 稳态实测（iters=10）：**19.5–19.6 ms/step = 123.3–123.6 GB/s**（spawn 版 113.7）
- 注：共享机上偶发 83 GB/s 读数，为同机 vllm/sglang 负载干扰
- [ ] bs>1 专家去重（T=8/E=256/K=6 下重复率仅 ~10%，优先级降低，大 batch 再做）

**单 token 解码（交互场景关键路径）**：
| 形状 | 延迟 | 带宽 | 全 CPU 估算 |
|---|---|---|---|
| DSV4 (H4096 I2048 k6, 43 层) | 3.7 ms/step | 81.8 GB/s | 3.7×43≈159 ms/tok ≈ 6 tok/s |
| GLM-5.2 (H6144 I2048 k8, 75 层) | 6.5 ms/step | 92.7 GB/s | 6.5×75≈488 ms/tok ≈ 2 tok/s |

对照 FreeToken 真机实测 31–32 / 16 tok/s：hybrid 后端 + GPU LRU 缓存承担大部分专家，CPU 只算 miss 的 ~82%。本内核作为 CPU 路径已具备支撑该吞吐的能力。

### ② core pinning ✅ 完成（结论反转：默认不 pin，opt-in）
实测矩阵（8tok DSV4 形状，双路 EPYC / 2 NUMA 节点 / 逻辑 CPU 交错编号）：
| 绑定策略 | 带宽 |
|---|---|
| **无 pinning（OS 调度）** | **110–113 GB/s** ✅ 最优 |
| taskset node0 (CPU0-31) | 103 GB/s |
| taskset node1 (CPU32-63) | 89 GB/s（远端内存） |
| shim 盲 pin 0–63（跨节点） | 57.9 GB/s ❌ |

结论：
- NUMA 感知调度交给 OS；固定绑定必须按拓扑设计（同节点 + 首触分配），盲绑有害
- 实现：`BLEND_PIN_CPU=<起始逻辑CPU>` 环境变量 opt-in，默认关闭
- FreeToken 的 "pin to cores 0..62" 在单路机器上等价于物理核集合，在双路上未必最优
- SMT 坑记录的"按物理核数配置线程数"结论仍然有效——那是线程数问题，与亲和性是两回事

### ② rayon 物理核 pinning 实装
- [ ] `ft-cli`/engine 启动时按 `physical_cores()` 配置线程池
- [ ] `core_affinity` crate 绑定工作线程到固定核集合
- [ ] mem-bench 默认线程数改为物理核数

### ③ cudarc 接入（为 Triton AOT cubin 铺路）
- [ ] 引入 cudarc（cuda-13 feature），与现有 raw FFI 并存对比
- [ ] stream 创建 + 异步 memcpy 替换同步版
- [ ] （远期）cuModuleLoad 加载 AOT cubin 的 PoC

### ④ engine 与 kernel 层对接
- [ ] `ft-engine` decode 步接入真实 MoE 执行路径（先 CPU 后 GPU）
- [ ] FTW 格式加载器（memmap 直达最终布局）

### ⑰ FlashInfer 接入 ✅ 编译通过，默认未启用（2026-08-23）
- [x] `kernels/basic/fi_single_decode.cu` 实例化 FI `SingleDecodeWithKVCacheDispatched<128, bf16>`
- [x] 编进 `libftkernels.so`（需 flashinfer headers + libcudacxx）
- [x] f32 实例化会踩 cp_async 512-bit 断言，已改 bf16
- [x] 端到端 `BLEND_FI=1` 可走 FI；默认仍 QKV+CPU softmax
- 短序列实测 FI 路径 8L ~56 tok/s < 默认 88（转换+逐 head RMSNorm 胶水更贵）
- 下一步要把 KV 直接存 bf16、合并 RMSNorm，FI 才有机会赢

### ⑯ GPU 融合注意力尝试 ✅ 结论：短序列回退（2026-08-23）
- [x] 实现 RMSNorm / RoPE / GQA decode / KV cache 的 CUDA 内核
- [x] 真机对比（seq≈16）：
  | | 8 层 | 48 层 |
  |---|---|---|
  | QKV 批处理 + CPU softmax | **88 / ~9 tok/s** | |
  | 全 GPU fused attn | 62 / 5.6 tok/s | 更慢 |
- 原因：每 head 一次 RMSNorm launch + softmax 单线程；短序列 kernel 启动 > 计算
- 决策：decode 默认仍用 QKV+CPU softmax；内核留在 libftkernels 供长上下文

### ⑮ 同模型基线：FreeToken vs blend（Qwen3-30B-A3B）✅ 完成（2026-08-23）
同卡 GPU4、同一份权重 `~/models/Qwen3-30B-A3B-Instruct`、都是 BF16：

| | 后端 | 注意力 | 稳态 decode |
|---|---|---|---|
| **FreeToken 0.1.2** | offload，cache=6144，CUDA Graph | FlashInfer | **111–132 tok/s** |
| **blend decode-qwen** | hybrid 384 槽 + AVX512 miss | GPU GEMV + CPU softmax | **~9 tok/s** |

差距约 **14×**。此模型上 **不是 FP4 的锅**（两边都是 BF16）。差在：
1. FlashInfer 融合注意力 vs 我们的 GEMV+CPU softmax
2. CUDA Graph 整步 vs 每层多次 sync/D2H
3. 专家在 GPU 上用他们的 fused MoE kernel，不是逐专家朴素 SwiGLU
4. 6144 槽几乎覆盖全部专家，miss 极少

### ⑭ QKV 合并发射 + q* 重叠 ✅ 完成（2026-08-23）
- [x] Q/K/V 一次 H2D、三个 GEMV、一次 sync（少 2 次同步）
- [x] GPU 专家 FFN 与 CPU miss 重叠（先 launch 再 moe_bf16 再 sync）
- [x] 真机：

  | | 之前 | **现在** |
  |---|---|---|
  | 8 层 | 57.8 tok/s | **88.1 tok/s** |
  | 48 层 | 9.3 tok/s | **8.6 tok/s**（波动，未再升） |
  | 8L prefill | 175 ms | **114 ms** |

### ⑬ decode-qwen 接 hybrid MoE ✅ 完成（2026-08-23）
- [x] 热专家上传并常驻 GpuSlotBank（384 槽）；miss 走 AVX512
- [x] 真机 Qwen3-30B 48 层（prompt=8, gen=8）：

  | 阶段 | tok/s |
  |---|---|
  | 全 CPU | 1.4 |
  | + GPU QKV/O | 6.6 |
  | **+ hybrid MoE** | **9.3** |
  | 8 层 hybrid | **57.8** |

- 整模已回到「专家-only CPU」的 9.3 tok/s，同时带上了真实注意力。

### ⑫ QKV/O 上 GPU ✅ 完成（2026-08-23）
- [x] `ft_gpu_gemv_bf16` 通用接口；decode-qwen 注意力投影常驻 VRAM
- [x] 真机对比（同一 Qwen3-30B，prompt=8, gen=4）：

  | | CPU 注意力 | **GPU GEMV** |
  |---|---|---|
  | 8 层 | 8.7 tok/s | **45.4 tok/s**（5.2×） |
  | 48 层 | 1.4 tok/s | **6.6 tok/s**（4.7×） |
  | 48L prefill 8 | 5.6 s | **1.14 s** |

- |h| 仍稳定。下一步：MoE 接 hybrid（热专家 GPU），冲击 10+ tok/s。

### ⑪ Qwen3 真实 Transformer decode ✅ 完成（2026-08-23）
- [x] RMSNorm + GQA（32/4 × d=128）+ Q/K RMSNorm + RoPE + KV cache
- [x] 专家 MoE 仍走 AVX512BF16；注意力/QKV 为 CPU f32 GEMV
- [x] 真机 Qwen3-30B-A3B：
  | | 加载 | prefill 8 | decode |
  |---|---|---|---|
  | 8 层 | 7.9 s | 896 ms | **114.7 ms/tok (8.7 tok/s)** |
  | **48 层** | 54 s | 5.6 s | **732 ms/tok (1.4 tok/s)** |
- |h| 稳定（84–288），不再消失；瓶颈已从专家切到朴素注意力 GEMV

### ⑩ 全模型专家栈 decode ✅ 完成（2026-08-23）
- [x] `blend moe-model`：48 层 × 128 专家全部装入（58 GB / 114 s）
- [x] 真实 router + AVX512BF16，attention stub = 残差直通
- [x] **108 ms/tok = 9.3 tok/s**（Qwen3-30B-A3B 专家路径，4 step 稳态）
- 无 attention/RMSNorm，数字是专家 FFN 下界；接上 attention 后还会更慢，hybrid+GPU 会再拉回去

### ⑨ 整层真实 MoE ✅ 完成（2026-08-23）
- [x] `blend moe-layer`：读 router + 128 专家（按分片批量 mmap），真实 top-8 softmax
- [x] Qwen3-30B-A3B 层 0，4 token：
  | | 延迟 | vs naive |
  |---|---|---|
  | naive f32 | 65.4 ms | — |
  | **AVX512BF16** | **7.6 ms** | **rel=3.5e-4 PASS**（8.6×） |
- 权重 1.2GB（w13 805 + w2 402）一次装入

### ⑧ 真实权重闭环 ✅ 完成（2026-08-23）
- [x] `ft-loader::locate_tensor` 读 HF `model.safetensors.index.json`
- [x] `blend real-expert --model Qwen3-30B-A3B --layer --expert`
- [x] 真机（层 0 专家 0 / 层 5 专家 17，BF16 H2048 I768）：
  | 对比 | rel |
  |---|---|
  | GPU vs naive | **1.0e-6 / 9.1e-7** PASS |
  | CPU SIMD vs naive | 2–3e-3（BF16 口径） |

### ⑦' GPU GEMV 优化 ✅ 完成（2026-08-23）
- [x] 激活进 smem（避免每行重读 x）+ 双 bf16 展开
- [x] DSV4 维度：**0.05 ms/expert，1111.7 GB/s**（接近 PRO 6000 HBM 量级）
- [x] 对拍不变 rel=1.5e-4 PASS
- 全 GPU 理论：43 层 × 6 专家 × 0.05 ms ≈ 13 ms/tok → ~77 tok/s

### ⑦ GPU 专家 FFN ✅ 完成（2026-08-23）
- [x] `ft_gpu_expert_ffn`：bf16 GEMV + silu_mul + axpy（CUDA）
- [x] `gpu-ffn-parity`：H4096/I2048 vs CPU naive **rel=1.5e-4 PASS**
- [x] hybrid `--real-pcie`：命中+Fetch 走 GPU FFN；warmup 后 GPU 45 ms / 4 tok ≈ 11 ms/tok
- 粗算：GPU 单专家 ~65µs vs CPU ~1.4ms（约 20×）；瓶颈仍是 CPU miss（~2 专家/层）

### ⑥ 真 PCIe H2D ✅ 完成（2026-08-23）
- [x] CUDA runtime：MemcpyAsync / Stream / MallocHost / SetDevice
- [x] `PinnedBuf` + `GpuSlotBank` + `Stream` RAII
- [x] `blend pcie-bench`（pinned 256MiB）：**57.7 GB/s，与 FreeToken benchbw 画像完全一致**
- [x] `hybrid-smoke --real-pcie`：Fetch 走真 DMA；单专家 pageable H2D 32.2 GB/s / 1.56 ms
- [x] 空闲卡验证：`CUDA_VISIBLE_DEVICES=4`

### ⑤'' HybridRuntime / DecodeDriver 收编 ✅ 完成（2026-08-23）
- [x] `ft-moe::HybridRuntime`：plan_layer + cpu_ids + reset_stats，sticky-routing 单测
- [x] `ft-engine::DecodeDriver<K: MoeKernel>`：按层 plan→compute，内核可注入
- [x] hybrid-smoke 改为驱动 DecodeDriver；warmup 后重置统计
- [x] 真机：命中率 **67.0%**（精确对齐 8/12）；tok/s 4.6–9.3（同机 vllm/sglang 争抢导致波动，架构正确）

### ⑤' LRU + q* hybrid 路径 ✅ 完成（2026-08-23）
- [x] `LruExpertCache`：(layer, expert) 键，3 个单测
- [x] `blend hybrid-smoke`：命中跳过 / Fetch 按 PCIe 计时 / CpuCompute 走真实 moe_bf16
- [x] 时间局部性模型（5/6 复用 + Fetch 填热集 + warmup）：
  | 指标 | 全 CPU (decode-smoke) | **hybrid-smoke** | FreeToken 实测 |
  |---|---|---|---|
  | 命中率 | 0% | **68%**（对齐论文 8/12） | ~67% |
  | tok/s | 6.3 | **15.7**（overlap 估算） | 31–32 |

剩余差距 15.7 → 31：GPU 命中在 VRAM 上算（~1TB/s，本冒烟计 0 成本偏乐观，但 CPU 仍是瓶颈）；
FreeToken CPU 内核 155 vs 我们 123 GB/s（×1.26 ≈ 20 tok/s），再加 GPU/CPU 真重叠与专家去重 → ~31。

### ⑤ engine decode 步接入真实 MoE ✅ 完成（2026-08-23）
- [x] `blend decode-smoke`：调度器 admit→prefill→decode 循环 + 每步 N 层真实 moe_bf16
- [x] 小形状冒烟：8 tok × 4 层 = 772 tok/s（调度器本身无瓶颈）
- [x] DSV4 全形状（43 层）：**6.3 tok/s / 159 ms/tok** —— 与 3.7ms×43 估算完全吻合
- 这是 **全 CPU、无 GPU LRU** 的下界；hybrid + GPU 缓存（FreeToken 实测 31–32 tok/s）是下一步接入 GPU 路径后的目标

## 五、已知风险 / 注意事项

1. **FreeToken csrc 许可**：Apache 2.0，抽取需保留版权声明与来源标注
2. **ISA 探测**：shim 必须运行时检测 AVX512BF16（`__builtin_cpu_supports("avx512bf16")`），否则 SIGILL
3. **线程模型冲突**：cpu_moe_ext 自带线程池 + pinning（63 线程）；与 blend 未来 rayon 方案二选一，
   先用它自带池验证性能，再决定是否替换成统一 rayon
4. **GPU 卡占用**：真机 GPU 0–3/6–7 有业务，测试一律 `CUDA_VISIBLE_DEVICES=4` 或 `=5`
5. **构建系统**：cuda feature 仅 Linux 可构建；build.rs 已用 CARGO_MANIFEST_DIR 绝对路径
