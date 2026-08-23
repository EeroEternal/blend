#!/bin/bash
# 构建 blend 内核库：
#   libftkernels.so — CUDA 内核（nvcc，需 CUDA 13；GPU FFI 路径）
#   libftcpu.so     — CPU AVX512BF16 MoE shim（g++，无 CUDA 依赖）
# 产物目录可经 FT_KERNELS_DIR 覆盖，默认 kernels/build。
set -euo pipefail
cd "$(dirname "$0")"
OUT="${FT_KERNELS_DIR:-$PWD/build}"
mkdir -p "$OUT"

# CUDA 部分（失败不阻塞 CPU 库——无 GPU 的机器仍可构建/测试 CPU 路径）
if command -v nvcc >/dev/null 2>&1; then
  ARCH=${FT_CUDA_ARCH:-sm_100}   # 默认 Blackwell (RTX PRO 6000)
  FI_INC=""
  # 常见 flashinfer 头文件位置（venv 或系统）
  for d in \
      ${FLASHINFER_INCLUDE:-} \
      "$HOME/freetoken-venv/lib/python3.10/site-packages/flashinfer/data/include" \
      /usr/local/include; do
    if [ -n "$d" ] && [ -f "$d/flashinfer/attention/decode.cuh" ]; then
      FI_INC="-I$d -DFT_HAS_FLASHINFER=1"
      break
    fi
  done
  SRC="basic/kernels.cu"
  if [ -n "$FI_INC" ]; then
    SRC="$SRC basic/fi_single_decode.cu"
    echo "flashinfer headers: $FI_INC"
  else
    echo "flashinfer headers not found; building without FI"
  fi
  nvcc -O3 --std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
       --generate-code=arch=compute_100,code=[compute_100,$ARCH] \
       -shared -Xcompiler -fPIC $FI_INC \
       $SRC -o "$OUT/libftkernels.so"
  echo "built $OUT/libftkernels.so"
else
  echo "skip libftkernels.so (nvcc not found)"
fi

# CPU 部分：ISA 由函数级 target attribute 控制，运行时探测分发
g++ -O3 -std=c++17 -fPIC -shared basic/cpu_moe_shim.cpp -o "$OUT/libftcpu.so" -lpthread
echo "built $OUT/libftcpu.so"
