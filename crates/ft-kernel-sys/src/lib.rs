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
        pub fn cudaMemcpy(
            dst: *mut std::ffi::c_void,
            src: *const std::ffi::c_void,
            count: usize,
            kind: c_int,
        ) -> FtStatus;
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
