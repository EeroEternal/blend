// FlashInfer single-decode 封装（Qwen3 GQA: 32/4, d=128, f32）
// 头文件来自 flashinfer 包 data/include。
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>

using flashinfer::DefaultAttention;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;
using flashinfer::SingleDecodeParams;
using flashinfer::SingleDecodeWithKVCacheDispatched;

// FlashInfer 的 MergeStates 对 f32 实例化会踩 cp_async 512-bit 断言；官方路径用 half/bf16。
extern "C" int ft_fi_single_decode(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                   const __nv_bfloat16* v, __nv_bfloat16* o, int heads,
                                   int kv_heads, int dim, int seq, int max_seq, void* stream) {
  if (dim != 128 || heads <= 0 || kv_heads <= 0 || seq <= 0 || heads % kv_heads != 0) {
    return -2;
  }
  using T = __nv_bfloat16;
  SingleDecodeParams<T, T, T> params(
      const_cast<T*>(q), const_cast<T*>(k), const_cast<T*>(v), o, nullptr,
      static_cast<uint32_t>(seq), static_cast<uint32_t>(heads),
      static_cast<uint32_t>(kv_heads), QKVLayout::kHND, 128, -1, 0.f,
      1.f / std::sqrt(128.f), 1.f, 1.0e4f);
  params.kv_stride_n = static_cast<uint32_t>(dim);
  params.kv_stride_h = static_cast<uint32_t>(max_seq * dim);
  params.kv_len = static_cast<uint32_t>(seq);

  using Variant = DefaultAttention<false, false, false, false>;
  cudaError_t st = SingleDecodeWithKVCacheDispatched<128, PosEncodingMode::kNone, Variant>(
      params, nullptr, static_cast<cudaStream_t>(stream));
  return st == cudaSuccess ? 0 : static_cast<int>(st);
}
