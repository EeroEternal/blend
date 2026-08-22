//! benchbw 标定：产出 q\* 硬件画像（对应 `ft bench bw`）。

use anyhow::Context;
use ft_moe::{BandwidthProfile, BackendKind, QStarPolicy};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 画像文件格式（对齐 freetoken benchbw.json 关键字段，可互操作）。
#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileFile {
    pub cpu_stream_gbps: f64,
    pub pcie_h2d_gbps: f64,
    pub threshold: f64,
    pub backend: String,
    pub fetch_fraction: f64,
}

/// 从已测带宽构造画像并保存。真实 STREAM/PCIe 微基准在 P2 接入 cudarc 后实现，
/// 当前接受外部测量值（如来自 FreeToken `ft bench bw` 的输出）。
pub fn save_profile(
    path: &Path,
    cpu_gbps: f64,
    pcie_gbps: f64,
) -> anyhow::Result<QStarPolicy> {
    let p = BandwidthProfile { cpu_gbps, pcie_gbps, threshold: 2.0 };
    let policy = QStarPolicy::calibrate(&p);
    let file = ProfileFile {
        cpu_stream_gbps: cpu_gbps,
        pcie_h2d_gbps: pcie_gbps,
        threshold: 2.0,
        backend: format!("{:?}", policy.backend()).to_lowercase(),
        fetch_fraction: policy.fetch_fraction(),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("create profile dir")?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&file)?)?;
    Ok(policy)
}

/// 读取画像。
pub fn load_profile(path: &Path) -> anyhow::Result<(BandwidthProfile, BackendKind)> {
    let raw = std::fs::read_to_string(path)?;
    let f: ProfileFile = serde_json::from_str(&raw)?;
    let backend = match f.backend.as_str() {
        "hybrid" => BackendKind::Hybrid,
        "offload" => BackendKind::Offload,
        "cpu" => BackendKind::Cpu,
        "fused" => BackendKind::Fused,
        other => anyhow::bail!("unknown backend in profile: {other}"),
    };
    Ok((
        BandwidthProfile { cpu_gbps: f.cpu_stream_gbps, pcie_gbps: f.pcie_h2d_gbps, threshold: f.threshold },
        backend,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_roundtrip() {
        let dir = std::env::temp_dir().join("blend-test-profile");
        let path = dir.join("benchbw.json");
        let pol = save_profile(&path, 124.4, 57.7).unwrap();
        assert_eq!(pol.backend(), BackendKind::Hybrid);
        let (p2, backend) = load_profile(&path).unwrap();
        assert_eq!(backend, BackendKind::Hybrid);
        assert!((p2.cpu_gbps - 124.4).abs() < 1e-9);
        let _ = std::fs::remove_file(&path);
    }
}
