# blend P1 真机验证记录（6000 Pro / EPYC 9355）

> 日期：2026-08-22
> 环境：8×RTX PRO 6000 Blackwell 工作站，AMD EPYC 9355（64 物理核/128 线程），
> 755GB DDR5，Ubuntu，Rust 1.97.1（Linux x86_64）
> 验证对象：commit `feat: mem-bench 多线程内存读带宽基准` 的 blend workspace

---

## 1. 跨平台构建与测试

| 项目 | macOS arm64（开发机） | Linux x86_64（EPYC 真机） |
|---|---|---|
| cargo test | 27/27 ✅ | **27/27 ✅** |

无平台相关编译错误；`ft-moe::policy` 的 x86 检测代码用 `is_x86_feature_detected`
运行时门控，aarch64 开发机可正常编译。

## 2. 数值对拍（parity）—— NaiveF32Executor vs torch

fixture 由服务器上 FreeToken venv 的 torch 生成（seed=42），
语义严格对齐执行器：`out = Σ_k rw·W2@(silu(W13g@x)·(W13u@x))`。

```
形状: tokens=4 hidden=64 inter=32 experts=8 k=2
max_abs_diff = 2.728e-12   (tol 1e-4)
PASS ✅
```

结论：CPU MoE 执行器的数学语义与 PyTorch 参考实现一致，
可作为后续 AVX512-BF16 内核（P3）的 golden 基线。

## 3. CPU MoE 吞吐基线（moe-bench）

| 形状 | 结果 |
|---|---|
| 小形状（cache 内，功能验证）：4tok H64 I32 E8 k2 | 16.2 GB/s |
| **DSV4 真实形状：8tok H4096 I2048 E256 k6** | **555.7 ms/step，8.7 GB/s** |

对照 FreeToken csrc AVX512BF16 内核同机实测 **155 GB/s**：

> naive f32 标量实现与目标内核有 **~18× 差距**。
> 这是 P3 SIMD 优化的量化空间与验收基线——按 DSV4 解码口径估算，
> 达到 155 GB/s 后单步 CPU MoE 计算约 31ms/8tok，支撑 ~30 tok/s 的解码路径。

## 4. 内存读带宽标定（mem-bench）

| 配置 | 读带宽 |
|---|---|
| 1 线程 | 43.7 GB/s |
| **64 线程（物理核数）** | **225.7 GB/s** |
| 128 线程（超线程满载） | 162.7 GB/s ⚠️ 超订反降 |

讨论：
- 与 FreeToken benchbw 实测的 124.4 GB/s 同量级；差异来自测量方法学
  （我们的口径为多线程顺序 u64 求和的**上界**，FreeToken 为 STREAM 标准测试）。
- 对 q\* 决策无影响：无论取 124 还是 226，cpu/pcie 比值均 > 2.0 阈值 → hybrid 后端不变。
- **经验教训**：线程数超过物理核数会因 SMT 争抢带宽反而下降（225→163），
  CPU MoE 执行器的 rayon 线程池应绑定物理核数（对齐 FreeToken 的 core pinning 做法）。

## 5. Linux 服务冒烟

```
GET  /v1/models           → {"data":[{"id":"blend-stub",...}]}        ✅
POST /v1/chat/completions → "linux smoke test [done]"                 ✅
```

## 6. 下一步（P2 入口）

1. `ft-parity` 扩展到 bf16 输入 + 大矩阵（H4096），确认 f32 累加误差仍在容差内
2. 在本机构建 `libftkernels.so`：fork FreeToken `kernel/csrc`（CUDA 13 工具链已就绪 `/usr/local/cuda-13.0`）
3. cudarc 接入：第一个真实 GPU kernel 调用走通 FFI 链路
4. rayon 线程池按物理核数配置 + core pinning（吸收本次 128t 教训）
