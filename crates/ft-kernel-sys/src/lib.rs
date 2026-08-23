#![allow(non_snake_case)]
//! libftkernels.so 的原始 FFI 绑定（架构文档 §5）。
//!
//! 内核来源：
//! 1. FreeToken `kernel/csrc` fork → CMake → libftkernels.so
//! 2. Triton kernel 经 FreeToken AOT 机制编译为 cubin，由 cudarc cuModuleLoad 加载（不在本 crate）
//! 3. flashinfer / sglang-kernel 稳定导出符号 dlopen

#[cfg(feature = "cuda")]
#[allow(non_upper_case_globals)]
    pub mod cuda {
    //! CUDA Runtime API 最小子集 + blend 自有内核入口。
    //! 链接目标：libcudart (CUDA 13) + libftkernels.so（kernels/build.sh 产物）
    use std::os::raw::{c_char, c_int};

    pub type FtStatus = c_int;
    pub const FT_OK: FtStatus = 0;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CudaDeviceProp {
        pub name: [c_char; 256],
        // 其余字段按 CUDA 文件布局占位；我们只用前 256 字节的名字。
        pub _pad: [u8; 8192 - 256],
    }

    extern "C" {
        // libcudart
        pub fn cudaGetDeviceCount(count: *mut c_int) -> FtStatus;
        pub fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: c_int) -> FtStatus;
        pub fn cudaMalloc(dev_ptr: *mut *mut std::ffi::c_void, size: usize) -> FtStatus;
        pub fn cudaFree(dev_ptr: *mut std::ffi::c_void) -> FtStatus;
        pub fn cudaMallocHost(ptr: *mut *mut std::ffi::c_void, size: usize) -> FtStatus;
        pub fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> FtStatus;
        pub fn cudaMemcpy(
            dst: *mut std::ffi::c_void,
            src: *const std::ffi::c_void,
            count: usize,
            kind: c_int,
        ) -> FtStatus;
        pub fn cudaMemcpyAsync(
            dst: *mut std::ffi::c_void,
            src: *const std::ffi::c_void,
            count: usize,
            kind: c_int,
            stream: *mut std::ffi::c_void,
        ) -> FtStatus;
        pub fn cudaStreamCreate(p: *mut *mut std::ffi::c_void) -> FtStatus;
        pub fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> FtStatus;
        pub fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> FtStatus;
        pub fn cudaSetDevice(device: c_int) -> FtStatus;
        pub fn cudaDeviceSynchronize() -> FtStatus;

        // libftkernels.so
        pub fn ft_cuda_device_count() -> c_int;
        pub fn ft_cuda_driver_version() -> c_int;
        pub fn ft_vector_add(
            a: *const f32,
            b: *const f32,
            out: *mut f32,
            n: c_int,
        ) -> c_int;
    }

    pub const cudaMemcpyHostToDevice: c_int = 1;
    pub const cudaMemcpyDeviceToHost: c_int = 2;
}

#[cfg(not(feature = "cuda"))]
pub mod ffi {
    /// 未启用 cuda feature：仅提供符号文档，不做链接。
    /// 所有调用方必须走 ft-kernel 的安全层并在运行时检查后端可用性。
    pub const _PLACEHOLDER: u8 = 0;
}

#[cfg(feature = "cpu-simd")]
pub mod cpusimd {
    //! CPU AVX512BF16 MoE shim（libftcpu.so，无 CUDA 依赖）。
    use std::os::raw::{c_char, c_int};

    extern "C" {
        pub fn ft_cpu_isa_name() -> *const c_char;
        /// bf16 MoE 前向；h [T,H] in/out f32，w13 [E,2I,H] w2 [E,H,I] bf16，
        /// ids [T,K]（负值跳过），rw [T,K]。返回 0 成功。
        pub fn ft_cpu_moe_bf16(
            h: *mut f32,
            w13: *const u16,
            w2: *const u16,
            ids: *const i32,
            rw: *const f32,
            t: c_int,
            hidden: c_int,
            inter: c_int,
            num_experts: c_int,
            k: c_int,
            threads: c_int,
        ) -> c_int;
    }
}
