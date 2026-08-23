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
    int block = 256;
    int grid = (n + block - 1) / block;
    vec_add_kernel<<<grid, block>>>(a, b, out, n);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

} // extern "C"

// ---- bf16 GEMV + SwiGLU（GPU 专家计算路径）----
__device__ __forceinline__ float d_bf16_f32(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    return __uint_as_float(u);
}

// out[row] = dot(w[row, 0:cols], x[0:cols])  w 为 bf16 行主
// x 搬进 smem，避免每个 output row 的 block 都从 HBM 重读激活。
__global__ void gemv_bf16_kernel(const uint16_t* w, const float* x, float* out,
                                 int rows, int cols) {
    extern __shared__ float xs[];
    for (int j = threadIdx.x; j < cols; j += blockDim.x) xs[j] = x[j];
    __syncthreads();
    int row = blockIdx.x;
    if (row >= rows) return;
    const uint16_t* wr = w + static_cast<size_t>(row) * cols;
    float acc = 0.f;
    // 一次走 2 个 bf16，提高指令密度
    int j = threadIdx.x * 2;
    const int stride = blockDim.x * 2;
    for (; j + 1 < cols; j += stride) {
        acc += d_bf16_f32(wr[j]) * xs[j];
        acc += d_bf16_f32(wr[j + 1]) * xs[j + 1];
    }
    if (j < cols) acc += d_bf16_f32(wr[j]) * xs[j];
    __shared__ float sm[256];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = sm[0];
}

// mid[i] = silu(gate[i]) * up[i]; gate=g[0:I], up=g[I:2I]
__global__ void silu_mul_kernel(const float* g, float* mid, int I) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < I) {
        float gate = g[i];
        float up = g[I + i];
        mid[i] = (gate / (1.f + expf(-gate))) * up;
    }
}

// y[i] += scale * x[i]
__global__ void axpy_kernel(float* y, const float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += scale * x[i];
}

extern "C" {

// 单专家 SwiGLU FFN：slot = [w13 (2I×H bf16) | w2 (H×I bf16)]
// x, y 为 device f32 [H]；y 累加 rw * down(silu(gate)*up)
int ft_gpu_expert_ffn(const uint16_t* slot, const float* x, float* y,
                      float* scratch_2i, float* scratch_i,
                      int H, int I, float rw, cudaStream_t stream) {
    const uint16_t* w13 = slot;
    const uint16_t* w2 = slot + static_cast<size_t>(2 * I) * H;
    const size_t smem_h = static_cast<size_t>(H) * sizeof(float);
    const size_t smem_i = static_cast<size_t>(I) * sizeof(float);
    gemv_bf16_kernel<<<I * 2, 256, smem_h, stream>>>(w13, x, scratch_2i, 2 * I, H);
    silu_mul_kernel<<<(I + 255) / 256, 256, 0, stream>>>(scratch_2i, scratch_i, I);
    // down: tmp_h[H] 复用 scratch_2i 前 H 个 float（H <= 2I 对 DSV4/GLM 成立）
    gemv_bf16_kernel<<<H, 256, smem_i, stream>>>(w2, scratch_i, scratch_2i, H, I);
    axpy_kernel<<<(H + 255) / 256, 256, 0, stream>>>(y, scratch_2i, rw, H);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int ft_gpu_zero(float* p, int n, cudaStream_t stream) {
    return cudaMemsetAsync(p, 0, static_cast<size_t>(n) * sizeof(float), stream) == cudaSuccess ? 0 : -1;
}

int ft_gpu_gemv_bf16(const uint16_t* w, const float* x, float* out,
                     int rows, int cols, cudaStream_t stream) {
    const size_t smem = static_cast<size_t>(cols) * sizeof(float);
    gemv_bf16_kernel<<<rows, 256, smem, stream>>>(w, x, out, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void rmsnorm_kernel(float* x, const float* w, int n, float eps) {
    __shared__ float ss;
    float local = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i] * x[i];
    __shared__ float sm[256];
    sm[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) ss = rsqrtf(sm[0] / n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] *= ss * w[i];
}

__global__ void rope_heads_kernel(float* x, int n_heads, int dim, int pos, float theta) {
    int hd = blockIdx.x;
    if (hd >= n_heads) return;
    float* h = x + hd * dim;
    int half = dim / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = pos * expf(-logf(theta) * (2.f * i) / dim);
        float s, c;
        sincosf(freq, &s, &c);
        float a = h[i], b = h[i + half];
        h[i] = a * c - b * s;
        h[i + half] = b * c + a * s;
    }
}

// GQA decode：每 block 一个 q head。K/V: [kv_heads, max_seq, dim]
__global__ void gqa_decode_kernel(const float* q, const float* k, const float* v,
                                  float* out, int heads, int kv_heads, int dim,
                                  int seq, int max_seq, float scale) {
    int hd = blockIdx.x;
    if (hd >= heads) return;
    int gqa = heads / kv_heads;
    int kv = hd / gqa;
    const float* qh = q + hd * dim;
    const float* kh = k + (static_cast<size_t>(kv) * max_seq) * dim;
    const float* vh = v + (static_cast<size_t>(kv) * max_seq) * dim;
    extern __shared__ float sh[];
    float* scores = sh; // seq
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        float s = 0.f;
        const float* kt = kh + t * dim;
        for (int d = 0; d < dim; ++d) s += qh[d] * kt[d];
        scores[t] = s * scale;
    }
    __syncthreads();
    // max
    if (threadIdx.x == 0) {
        float mx = -1e30f;
        for (int t = 0; t < seq; ++t) mx = fmaxf(mx, scores[t]);
        float sum = 0.f;
        for (int t = 0; t < seq; ++t) {
            scores[t] = expf(scores[t] - mx);
            sum += scores[t];
        }
        float inv = 1.f / sum;
        for (int t = 0; t < seq; ++t) scores[t] *= inv;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float o = 0.f;
        for (int t = 0; t < seq; ++t) o += scores[t] * vh[t * dim + d];
        out[hd * dim + d] = o;
    }
}

extern "C" {

int ft_gpu_rmsnorm(float* x, const float* w, int n, float eps, cudaStream_t stream) {
    rmsnorm_kernel<<<1, 256, 0, stream>>>(x, w, n, eps);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int ft_gpu_rope(float* x, int n_heads, int dim, int pos, float theta, cudaStream_t stream) {
    rope_heads_kernel<<<n_heads, 64, 0, stream>>>(x, n_heads, dim, pos, theta);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int ft_gpu_gqa_decode(const float* q, const float* k, const float* v, float* out,
                      int heads, int kv_heads, int dim, int seq, int max_seq,
                      float scale, cudaStream_t stream) {
    size_t smem = static_cast<size_t>(seq) * sizeof(float);
    gqa_decode_kernel<<<heads, 128, smem, stream>>>(q, k, v, out, heads, kv_heads, dim, seq, max_seq, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int ft_gpu_copy_kv(float* cache, const float* src, int kv_heads, int dim, int pos, int max_seq,
                   cudaStream_t stream) {
    // src [kv_heads, dim] -> cache[kv, pos, dim]
    for (int kv = 0; kv < kv_heads; ++kv) {
        float* dst = cache + (static_cast<size_t>(kv) * max_seq + pos) * dim;
        const float* s = src + kv * dim;
        cudaMemcpyAsync(dst, s, dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    return 0;
}

} // extra extern

} // extern "C"
