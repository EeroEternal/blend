//! Decode 驱动：调度器选出 Decode 步后，按层走 HybridRuntime。
//!
//! 具体 MoE 计算通过 `MoeKernel` trait 注入（SIMD / naive / stub），
//! 本模块不依赖 CUDA 或 libftcpu。

use ft_moe::{HybridRuntime, MoePlan, QStarPolicy};

/// 一层 MoE 计算的注入点。
pub trait MoeKernel {
    fn compute(&mut self, plan: &MoePlan, layer: u32);
}

/// 把 hybrid 决策接到任意内核。
pub struct DecodeDriver<K: MoeKernel> {
    pub rt: HybridRuntime,
    pub kernel: K,
    pub layers: usize,
}

impl<K: MoeKernel> DecodeDriver<K> {
    pub fn new(cache_slots: usize, q: QStarPolicy, kernel: K, layers: usize) -> Self {
        Self { rt: HybridRuntime::new(cache_slots, q), kernel, layers }
    }

    /// 对每一层执行：plan → kernel.compute。
    /// `router` 给出该层的 routed expert id 列表。
    pub fn decode_token(&mut self, mut router: impl FnMut(u32) -> Vec<u32>) {
        for layer in 0..self.layers {
            let routed = router(layer as u32);
            let plan = self.rt.plan_layer(layer as u32, &routed);
            self.kernel.compute(&plan, layer as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ft_moe::BandwidthProfile;

    struct CountingKernel {
        calls: usize,
    }
    impl MoeKernel for CountingKernel {
        fn compute(&mut self, _plan: &MoePlan, _layer: u32) {
            self.calls += 1;
        }
    }

    #[test]
    fn driver_invokes_kernel_per_layer() {
        let q = QStarPolicy::calibrate(&BandwidthProfile::pro6000_epyc9355());
        let mut d = DecodeDriver::new(32, q, CountingKernel { calls: 0 }, 4);
        d.decode_token(|l| vec![l, l + 1]);
        assert_eq!(d.kernel.calls, 4);
        assert!(d.rt.stats().total() > 0);
    }
}
