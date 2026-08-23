// FlashInfer single-decode 封装（Qwen3 GQA: 32/4, d=128, f32）
// 头文件来自 flashinfer 包 data/include。
#include <cmath>
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

extern "C" int ft_fi_single_decode(const float* q, const float* k, const float* v, float* o,
                                   int heads, int kv_heads, int dim, int seq, int max_seq,
                                   void* stream) {
  if (dim != 128 || heads <= 0 || kv_heads <= 0 || seq <= 0 || heads % kv_heads != 0) {
    return -2;
  }
  SingleDecodeParams<float, float, float> params(
      const_cast<float*>(q), const_cast<float*>(k), const_cast<float*>(v), o,
      /*alibi*/ nullptr, static_cast<uint32_t>(seq), static_cast<uint32_t>(heads),
      static_cast<uint32_t>(kv_heads), QKVLayout::kHND, 128, /*window_left*/ -1,
      /*logits_soft_cap*/ 0.f, 1.f / std::sqrt(128.f), /*rope_scale*/ 1.f,
      /*rope_theta*/ 1.0e4f);
  // 缓存按 max_seq 分配，覆盖 constructor 里用 seq 当 stride 的假设
  params.kv_stride_n = static_cast<uint32_t>(dim);
  params.kv_stride_h = static_cast<uint32_t>(max_seq * dim);
  params.kv_len = static_cast<uint32_t>(seq);

  using Variant = DefaultAttention<false, false, false, false>;
  cudaError_t st = SingleDecodeWithKVCacheDispatched<128, PosEncodingMode::kNone, Variant>(
      params, /*tmp*/ nullptr, static_cast<cudaStream_t>(stream));
  return st == cudaSuccess ? 0 : static_cast<int>(st);
}
