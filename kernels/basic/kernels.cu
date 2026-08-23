// blend 基础内核库（libftkernels.so 的第一个成员）。
// 后续按 P2 计划逐步吸收 FreeToken csrc 内核（nvfp4 MoE / DSA attention 等）。
#include <cuda_runtime.h>
#include <cstdint>

__global__ void vec_add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" {

int ft_cuda_device_count() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return e == cudaSuccess ? n : -1;
}

int ft_cuda_driver_version() {
    int v = 0;
    cudaDriverGetVersion(&v);
    return v;
}

const char* ft_cuda_last_error() {
    return cudaGetErrorString(cudaPeekAtLastError());
}

int ft_vector_add(const float* a, const float* b, float* out, int n) {
    // a/b/out 为 device 指针
    int block = 256;
    int grid = (n + block - 1) / block;
    vec_add_kernel<<<grid, block>>>(a, b, out, n);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

} // extern "C"
