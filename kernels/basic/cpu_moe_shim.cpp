// blend cpu MoE shim — AVX512BF16 GEMV 内核
// 来源：FreeToken (https://github.com/FlashML-org/FreeToken)
//       python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp（Apache-2.0）
//       抽取其纯计算函数段（bf16 转换 / dot 函数族），剥离 torch/pybind 依赖。
// 语义对齐 crates/ft-moe/src/cpu.rs::NaiveF32Executor：
//   h[t] = Σ_k rw[t,k] * Down_e( silu(gate@x) * (up@x) )
// 数值口径：权重与激活走 BF16、f32 累加（对齐 FreeToken 生产路径）。
#include <cstdint>
#include <cstring>
#include <cmath>
#include <atomic>
#include <thread>
#include <vector>
#include <cstdlib>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#define SHIM_HAS_AFFINITY 1
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

using bf16_t = uint16_t;

inline float bf16_to_f32(bf16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bf16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16_t>(u >> 16);
}

inline float act_silu(float x) { return x / (1.0f + std::exp(-x)); }

constexpr int PF_AHEAD = 512;  // 权重流预取距离（字节）

using dot_fn = float (*)(const bf16_t*, const bf16_t*, int);

float dot_scalar(const bf16_t* w, const bf16_t* x, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return acc;
}

#if defined(__x86_64__)
__attribute__((target("avx512f")))
float dot_avx512f(const bf16_t* w, const bf16_t* x, int n) {
  // 4 独立累加器提升内存级并行（来源同上）
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 64; j += 16) {
      __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + j));
      __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + j));
      __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
      __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
      __m512& acc = (j == 0) ? a0 : (j == 16) ? a1 : (j == 32) ? a2 : a3;
      acc = _mm512_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 16 <= n; i += 16) {
    __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
    __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
    __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
    a0 = _mm512_fmadd_ps(wf, xf, a0);
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

__attribute__((target("avx512bf16,avx512f")))
static inline __m512bh load_bh(const bf16_t* p) {
  __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
  __m512bh out;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

__attribute__((target("avx512bf16,avx512f")))
float dot_avx512bf16(const bf16_t* w, const bf16_t* x, int n) {
  // 4 累加器 × 128 bf16/迭代：内存带宽受限 GEMV 的关键设计（来源同上）
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
    a1 = _mm512_dpbf16_ps(a1, load_bh(w + i + 32), load_bh(x + i + 32));
    a2 = _mm512_dpbf16_ps(a2, load_bh(w + i + 64), load_bh(x + i + 64));
    a3 = _mm512_dpbf16_ps(a3, load_bh(w + i + 96), load_bh(x + i + 96));
  }
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // __x86_64__

dot_fn select_dot() {
#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx512bf16")) return dot_avx512bf16;
  if (__builtin_cpu_supports("avx512f")) return dot_avx512f;
#endif
  return dot_scalar;
}

const char* isa_name_of(dot_fn fn) {
#if defined(__x86_64__)
  if (fn == dot_avx512bf16) return "avx512bf16";
  if (fn == dot_avx512f) return "avx512f";
#endif
  return "scalar";
}

}  // namespace

extern "C" {

/// 返回检测到的 ISA 名。
const char* ft_cpu_isa_name() { return isa_name_of(select_dot()); }

// 工作线程亲和性：默认关闭（见 docs/worklog-status.md 风险与实测结论：
// 双路 EPYC 上 OS 自由调度 110-113 GB/s 优于任何固定绑定，跨 NUMA 绑定最差 57.9）。
// opt-in：环境变量 BLEND_PIN_CPU=<n> 时，工作线程 i pin 到逻辑 CPU (n+i)。
static bool pin_enabled() {
#if SHIM_HAS_AFFINITY
  static const bool on = [] {
    const char* e = std::getenv("BLEND_PIN_CPU");
    return e != nullptr && e[0] != '\0';
  }();
  return on;
#else
  return false;
#endif
}

static int pin_base() {
#if SHIM_HAS_AFFINITY
  const char* e = std::getenv("BLEND_PIN_CPU");
  return e ? std::atoi(e) : 0;
#else
  return 0;
#endif
}

static void pin_to(int idx) {
#if SHIM_HAS_AFFINITY
  if (!pin_enabled()) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(pin_base() + idx, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)idx;
#endif
}

/// bf16 MoE 前向。返回 0 成功，非 0 参数错误。
/// 布局：w13 [E,2I,H]、w2 [E,H,I]（行主）、ids [T,K]（负值跳过）、rw [T,K]、h [T,H] in/out。
/// 两阶段行级并行：pass1 按 (t,j,row) 算 gate/up；激活融合；pass2 按 (t,o) 行算输出，
/// 每个输出元素只被一个线程写，无竞争。线程按调用创建（常驻 decode 路径后续换持久池，
/// 见 docs/worklog-status.md 风险 3）。
int ft_cpu_moe_bf16(float* h, const uint16_t* w13, const uint16_t* w2,
                    const int32_t* ids, const float* rw,
                    int T, int H, int I, int E, int K, int threads) {
  if (T <= 0 || H <= 0 || I <= 0 || E <= 0 || K <= 0 || threads <= 0) return -1;

  // 激活转 bf16（一次）
  std::vector<bf16_t> xb(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < xb.size(); ++i) xb[i] = f32_to_bf16(h[i]);

  // 中间层：pass1 存 gate/up 原始点积到 g [T,K,2I]；融合后写 mid [T,K,I]
  std::vector<bf16_t> g(static_cast<size_t>(T) * K * 2 * I);
  std::vector<bf16_t> mid(static_cast<size_t>(T) * K * I);

  const dot_fn dot = select_dot();
  const int nth = threads > 1 ? threads : 1;

  auto pass1 = [&]() {
    const int64_t per = 2LL * I;
    const int64_t total = static_cast<int64_t>(T) * K * per;
    std::atomic<int64_t> next{0};
    auto body = [&]() {
      for (;;) {
        const int64_t task = next.fetch_add(1);
        if (task >= total) break;
        const int t = static_cast<int>(task / (static_cast<int64_t>(K) * per));
        const int64_t rem = task - static_cast<int64_t>(t) * K * per;
        const int j = static_cast<int>(rem / per);
        const int row = static_cast<int>(rem % per);
        const int e = ids[static_cast<size_t>(t) * K + j];
        if (e < 0 || e >= E) continue;
        const bf16_t* wrow = w13 + (static_cast<size_t>(e) * 2 * I + row) * H;
        const float v = dot(wrow, xb.data() + static_cast<size_t>(t) * H, H);
        g[static_cast<size_t>(t) * K * per + static_cast<size_t>(j) * per + row] =
            f32_to_bf16(v);
      }
    };
    std::vector<std::thread> ths;
    ths.reserve(nth - 1);
    for (int i = 1; i < nth; ++i)
      ths.emplace_back([&, i] { pin_to(i); body(); });
    body();
    for (auto& th : ths) th.join();
  };

  pass1();

  // 激活融合：mid[t,j,i] = silu(gate) * up
  // 布局：g 内 (t,j) 块占 2I —— gate 在 [base+i]，up 在 [base+I+i]
  for (int ti = 0; ti < T; ++ti) {
    for (int j = 0; j < K; ++j) {
      const bf16_t* src = &g[(static_cast<size_t>(ti) * K + j) * 2 * I];
      bf16_t* dst = &mid[(static_cast<size_t>(ti) * K + j) * I];
      for (int i = 0; i < I; ++i) {
        dst[i] = f32_to_bf16(act_silu(bf16_to_f32(src[i])) * bf16_to_f32(src[I + i]));
      }
    }
  }

  auto pass2 = [&]() {
    const int64_t total = static_cast<int64_t>(T) * H;
    std::atomic<int64_t> next{0};
    auto body = [&]() {
      for (;;) {
        const int64_t task = next.fetch_add(1);
        if (task >= total) break;
        const int t = static_cast<int>(task / H);
        const int o = static_cast<int>(task % H);
        float acc = 0.0f;
        const bf16_t* gv = mid.data() + static_cast<size_t>(t) * K * I;
        for (int j = 0; j < K; ++j) {
          const int e = ids[static_cast<size_t>(t) * K + j];
          if (e < 0 || e >= E) continue;
          const float wgt = rw[static_cast<size_t>(t) * K + j];
          if (wgt == 0.0f) continue;
          const bf16_t* wrow = w2 + (static_cast<size_t>(e) * H + o) * I;
          acc += wgt * dot(wrow, gv + static_cast<size_t>(j) * I, I);
        }
        h[static_cast<size_t>(t) * H + o] = acc;
      }
    };
    std::vector<std::thread> ths;
    ths.reserve(nth - 1);
    for (int i = 1; i < nth; ++i)
      ths.emplace_back([&, i] { pin_to(i); body(); });
    body();
    for (auto& th : ths) th.join();
  };

  pass2();
  return 0;
}

}  // extern "C"
