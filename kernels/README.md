# blend kernels

GPU 内核层（架构文档 §5）。当前阶段（P0/P1）不需要构建本目录。

## 计划内容

1. `csrc/` — 从 FreeToken `python/freetoken/kernel/csrc` fork 的 CUDA/C++ 内核
   （cpu_moe AVX512BF16、pinned_tensor、jit 基础设施），CMake 构建 → `libftkernels.so`
2. `aot/` — Triton 内核 AOT 编译脚本（nvfp4/fp8 fused MoE、DSA sparse attention、RoPE 等）
   产出 cubin 集合，Rust 侧经 cudarc `cuModuleLoad` 加载
3. `parity/` — 数值对拍工具：同输入喂 Python 引擎与本引擎，逐层 max-abs-diff < 1e-2

## 构建要求

- CUDA 13 toolkit（nvcc）
- `FT_KERNELS_DIR` 环境变量指向构建产物目录，供 `ft-kernel-sys`（feature = "cuda"）链接
