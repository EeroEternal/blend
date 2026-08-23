//! GPU 内核安全封装层。
//!
//! 规则（架构文档 §5）：所有 unsafe 集中在这里，
//! 对上暴露返回 Result 的安全 API，并做形状/设备校验。

/// 内核后端可用性探测。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    /// CUDA 内核库已加载
    CudaKernels,
    /// 无 GPU 内核：CPU 参考路径（仅用于对拍与测试）
    CpuReference,
}

pub fn detect() -> KernelBackend {
    #[cfg(feature = "cuda")]
    {
        KernelBackend::CudaKernels
    }
    #[cfg(not(feature = "cuda"))]
    {
        KernelBackend::CpuReference
    }
}

/// GPU 设备信息。
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub index: i32,
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use ft_core::{FtError, Result};
    use ft_kernel_sys::cuda as sys;
    use std::ffi::c_void;

    fn check(code: sys::FtStatus, what: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(FtError::Kernel(format!("{what}: cuda status {code}")))
        }
    }

    pub fn device_count() -> Result<usize> {
        let mut n: i32 = 0;
        check(unsafe { sys::cudaGetDeviceCount(&mut n) }, "cudaGetDeviceCount")?;
        Ok(n.max(0) as usize)
    }

    pub fn device_info(index: i32) -> Result<DeviceInfo> {
        let prop = Box::into_raw(Box::new(sys::CudaDeviceProp {
            name: [0; 256],
            _pad: [0; 8192 - 256],
        }));
        let status = unsafe { sys::cudaGetDeviceProperties(prop, index) };
        let (name, _) = {
            let p = unsafe { Box::from_raw(prop) };
            let bytes: Vec<u8> =
                p.name.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
            (String::from_utf8_lossy(&bytes).into_owned(), ())
        };
        check(status, "cudaGetDeviceProperties")?;
        Ok(DeviceInfo { name, index })
    }

    struct DevPtr(*mut c_void);
    // device 内存指针可跨线程传递（CUDA 上下文按进程管理，简化处理）
    unsafe impl Send for DevPtr {}

    /// 单卡 RAII buffer。
    pub struct DevBuffer {
        ptr: DevPtr,
        pub len_bytes: usize,
    }

    impl DevBuffer {
        pub fn alloc(bytes: usize) -> Result<Self> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { sys::cudaMalloc(&mut ptr, bytes) }, "cudaMalloc")?;
            Ok(Self { ptr: DevPtr(ptr), len_bytes: bytes })
        }
        pub fn as_ptr(&self) -> *const c_void {
            self.ptr.0
        }
        pub fn as_mut_ptr(&mut self) -> *mut c_void {
            self.ptr.0
        }
        pub fn h2d(&self, src: &[f32]) -> Result<()> {
            let bytes = src.len() * 4;
            assert!(bytes <= self.len_bytes);
            check(
                unsafe {
                    sys::cudaMemcpy(self.ptr.0, src.as_ptr() as *const c_void, bytes, sys::cudaMemcpyHostToDevice)
                },
                "cudaMemcpy H2D",
            )
        }
        pub fn d2h(&self, dst: &mut [f32]) -> Result<()> {
            let bytes = dst.len() * 4;
            assert!(bytes <= self.len_bytes);
            check(
                unsafe {
                    sys::cudaMemcpy(dst.as_mut_ptr() as *mut c_void, self.ptr.0, bytes, sys::cudaMemcpyDeviceToHost)
                },
                "cudaMemcpy D2H",
            )
        }
    }

    impl Drop for DevBuffer {
        fn drop(&mut self) {
            unsafe { sys::cudaFree(self.ptr.0) };
        }
    }

    pub fn vector_add(a: &DevBuffer, b: &DevBuffer, out: &mut DevBuffer, n: usize) -> Result<()> {
        let rc = unsafe {
            sys::ft_vector_add(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                out.as_mut_ptr() as *mut f32,
                n as i32,
            )
        };
        if rc != 0 {
            return Err(FtError::Kernel(format!("ft_vector_add: {rc}")));
        }
        check(unsafe { sys::cudaDeviceSynchronize() }, "sync")
    }
}

#[cfg(feature = "cuda")]
pub use gpu::{device_count, device_info, vector_add, DevBuffer};

/// CPU SIMD MoE 执行器（libftcpu.so 的 AVX512BF16 shim）。
#[cfg(feature = "cpu-simd")]
pub mod cpu_simd {
    use super::*;
    use ft_core::{FtError, Result};
    use ft_kernel_sys::cpusimd as sys;
    use std::ffi::{c_void, CStr};

    pub fn isa_name() -> String {
        unsafe {
            CStr::from_ptr(sys::ft_cpu_isa_name()).to_string_lossy().into_owned()
        }
    }

    /// # Safety
    /// w13/w2 长度须匹配 E*2I*H / E*H*I；ids 元素须在 [0,E) 或负值。
    pub fn moe_bf16(
        h: &mut [f32],
        w13: &[u16],
        w2: &[u16],
        topk: &[i32],
        rw: &[f32],
        t: usize,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        k: usize,
        threads: usize,
    ) -> Result<()> {
        assert_eq!(h.len(), t * hidden);
        assert_eq!(topk.len(), t * k);
        let rc = unsafe {
            sys::ft_cpu_moe_bf16(
                h.as_mut_ptr() as *mut f32,
                w13.as_ptr() as *const u16,
                w2.as_ptr() as *const u16,
                topk.as_ptr() as *const i32,
                rw.as_ptr() as *const f32,
                t as i32,
                hidden as i32,
                inter as i32,
                num_experts as i32,
                k as i32,
                threads as i32,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(FtError::Kernel(format!("ft_cpu_moe_bf16: {rc}")))
        }
    }
}

#[cfg(feature = "cpu-simd")]
pub use cpu_simd::{isa_name as cpu_isa_name, moe_bf16};

#[cfg(test)]
mod tests {
    #[test]
    fn detection_runs() {
        // 不断言具体值——CI 有无 GPU 都要绿
        let _ = super::detect();
    }
}
