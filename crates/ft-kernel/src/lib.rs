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

#[cfg(test)]
mod tests {
    #[test]
    fn detection_runs() {
        // 不断言具体值——CI 有无 GPU 都要绿
        let _ = super::detect();
    }
}
