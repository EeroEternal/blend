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

### ① AVX512BF16 内核吸收 + 对拍（当前任务）
- [ ] 抽取计算函数段到 `kernels/basic/cpu_moe_shim.cpp`（记录来源 commit hash）
- [ ] build.sh 增加 CPU 源编译（`-march=nocona -mavx512f -mavx512bf16 -mavx512vl`）
- [ ] `ft-kernel-sys` 增加 `ft_cpu_moe_bf16` 声明
- [ ] `blend parity --kernel avx512`：与 NaiveF32Executor/torch 三方对拍（tol 1e-2 相对，BF16 精度口径）
- [ ] `blend moe-bench --kernel avx512`：验证吞吐从 8.7 → ~155 GB/s
- 验收标准：DSV4 形状下 ≥100 GB/s，且三方 diff 在 BF16 容差内

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

## 五、已知风险 / 注意事项

1. **FreeToken csrc 许可**：Apache 2.0，抽取需保留版权声明与来源标注
2. **ISA 探测**：shim 必须运行时检测 AVX512BF16（`__builtin_cpu_supports("avx512bf16")`），否则 SIGILL
3. **线程模型冲突**：cpu_moe_ext 自带线程池 + pinning（63 线程）；与 blend 未来 rayon 方案二选一，
   先用它自带池验证性能，再决定是否替换成统一 rayon
4. **GPU 卡占用**：真机 GPU 0–3/6–7 有业务，测试一律 `CUDA_VISIBLE_DEVICES=4` 或 `=5`
5. **构建系统**：cuda feature 仅 Linux 可构建；build.rs 已用 CARGO_MANIFEST_DIR 绝对路径
