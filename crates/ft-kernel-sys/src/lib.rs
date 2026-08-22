#![allow(non_snake_case)]
//! libftkernels.so 的原始 FFI 绑定（架构文档 §5）。
//!
//! 内核来源：
//! 1. FreeToken `kernel/csrc` fork → CMake → libftkernels.so
//! 2. Triton kernel 经 FreeToken AOT 机制编译为 cubin，由 cudarc cuModuleLoad 加载（不在本 crate）
//! 3. flashinfer / sglang-kernel 稳定导出符号 dlopen

#[cfg(feature = "cuda")]
pub mod ffi {
    use std::os::raw::{c_int, c_void};

    // 张量一律以 (ptr, device_ptr) 裸指针传递；stream 为 cudaStream_t。
    pub type FtStream = *mut c_void;
    pub type FtStatus = c_int;

    pub const FT_OK: FtStatus = 0;

    extern "C" {
        /// NVFP4 fused MoE（gate/up/down + silu + 路由加权）
        pub fn ft_nvfp4_fused_moe(
            hidden: *mut c_void,      // [tokens, H] bf16 in/out
            w13: *const c_void,       // [E, 2I, H] nvfp4
            w2: *const c_void,        // [E, H, I] nvfp4
            topk: *const u32,         // [tokens, k]
            weights: *const f32,      // [tokens, k]
            tokens: c_int,
            num_experts: c_int,
            hidden_size: c_int,
            inter_size: c_int,
            k: c_int,
            stream: FtStream,
        ) -> FtStatus;

        /// DSA sparse attention（DeepSeek sparse attention index+gather）
        pub fn ft_dsa_sparse_attention(
            q: *const c_void,
            kv_cache: *mut c_void,
            indexer: *const c_void,
            out: *mut c_void,
            stream: FtStream,
        ) -> FtStatus;

        /// 批量 pinned→device memcpy（prefill 双缓冲的取回路径）
        pub fn ft_batch_memcpy_async(
            dsts: *const *mut c_void,
            srcs: *const *const c_void,
            sizes: *const usize,
            n: c_int,
            stream: FtStream,
        ) -> FtStatus;
    }
}

#[cfg(not(feature = "cuda"))]
pub mod ffi {
    /// 未启用 cuda feature：仅提供符号文档，不做链接。
    /// 所有调用方必须走 ft-kernel 的安全层并在运行时检查后端可用性。
    pub const _PLACEHOLDER: u8 = 0;
}
