//! q\* 带宽自适应策略（架构文档 §3 / FreeToken 论文 §3.2）。
//!
//! 纯函数、无 IO、无 GPU 依赖 —— 这是 fork 后最常调的模块，
//! 所有决策可单测、可用 criterion 基准锁定。

/// benchbw 标定出的硬件画像（对应 ~/.cache/freetoken/benchbw.json 的核心字段）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandwidthProfile {
    /// CPU 内存有效读带宽 GB/s
    pub cpu_gbps: f64,
    /// PCIe H2D 有效带宽 GB/s
    pub pcie_gbps: f64,
    /// hybrid 推荐阈值：cpu/pcie 超过此值才启用 hybrid
    pub threshold: f64,
}

impl BandwidthProfile {
    /// 参考值来自 2026-08-22 PRO 6000 8 卡机实测：
    /// CPU STREAM 124.4 GB/s，PCIe H2D 57.7 GB/s
    pub fn pro6000_epyc9355() -> Self {
        Self { cpu_gbps: 124.4, pcie_gbps: 57.7, threshold: 2.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Fused,
    Offload,
    Cpu,
    Hybrid,
}

/// 由画像决定后端（对齐 `ft bench bw` 的 decision rule）。
pub fn recommend_backend(p: &BandwidthProfile) -> BackendKind {
    let ratio = p.cpu_gbps / p.pcie_gbps;
    if ratio > p.threshold {
        BackendKind::Hybrid
    } else {
        BackendKind::Offload
    }
}

/// q\* 策略。
#[derive(Debug, Clone, Copy)]
pub struct QStarPolicy {
    profile: BandwidthProfile,
    /// 每步 miss 中走 PCIe 取回的比例上限（标定输出）
    fetch_fraction: f64,
}

impl QStarPolicy {
    /// 从画像标定。fetch_fraction = B_pcie / (B_pcie + B_cpu_overlap)，
    /// 其中 B_cpu_overlap 是与 PCIe 重叠时 CPU 可维持的带宽。
    /// 对齐实测口径：overlap 时 pcie≈34GB/s、cpu≈148GB/s → fetch ≈ 18.7%。
    pub fn calibrate(p: &BandwidthProfile) -> Self {
        // 用线性带宽模型近似实测重叠行为
        let overlap_pcie = p.pcie_gbps * 0.6; // 随机 gather 折损
        let overlap_cpu = p.cpu_gbps * 1.19; // 与论文一致：重叠时 CPU 反而更高效
        let fetch_fraction = overlap_pcie / (overlap_pcie + overlap_cpu);
        Self { profile: *p, fetch_fraction }
    }

    pub fn fetch_fraction(&self) -> f64 {
        self.fetch_fraction
    }

    pub fn backend(&self) -> BackendKind {
        recommend_backend(&self.profile)
    }

    /// 本步 m 个 miss 的拆分基数：取回多少个专家。
    pub fn fetch_count(&self, misses: usize) -> usize {
        (misses as f64 * self.fetch_fraction).round() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pro6000_recommends_hybrid() {
        let p = BandwidthProfile::pro6000_epyc9355();
        assert_eq!(recommend_backend(&p), BackendKind::Hybrid);
    }

    #[test]
    fn balanced_machine_recommends_offload() {
        let p = BandwidthProfile { cpu_gbps: 80.0, pcie_gbps: 55.0, threshold: 2.0 };
        assert_eq!(recommend_backend(&p), BackendKind::Offload);
    }

    #[test]
    fn fetch_fraction_near_measured_18_percent() {
        let pol = QStarPolicy::calibrate(&BandwidthProfile::pro6000_epyc9355());
        // 实测 18.4%（dsv4）/18.7%（nvfp4），允许模型误差 ±3pp
        assert!((pol.fetch_fraction() - 0.187).abs() < 0.03, "got {}", pol.fetch_fraction());
    }

    #[test]
    fn fetch_count_sane() {
        let pol = QStarPolicy::calibrate(&BandwidthProfile::pro6000_epyc9355());
        assert_eq!(pol.fetch_count(12), 2); // 12 * ~0.187 ≈ 2.24 → 2
        assert_eq!(pol.fetch_count(0), 0);
        assert_eq!(pol.fetch_count(100), 19);
    }
}
