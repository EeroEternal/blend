#!/bin/bash
# 构建 libftkernels.so。要求: nvcc (CUDA 13) 在 PATH。
# 产物目录可经 FT_KERNELS_DIR 覆盖，默认 kernels/build。
set -euo pipefail
cd "$(dirname "$0")"
OUT="${FT_KERNELS_DIR:-$PWD/build}"
mkdir -p "$OUT"
ARCH=${FT_CUDA_ARCH:-sm_100}   # 默认 Blackwell (RTX PRO 6000)
nvcc -O3 --generate-code=arch=compute_100,code=[compute_100,$ARCH] \
     -shared -Xcompiler -fPIC \
     basic/kernels.cu -o "$OUT/libftkernels.so"
echo "built $OUT/libftkernels.so"
