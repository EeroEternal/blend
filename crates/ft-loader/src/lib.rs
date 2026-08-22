//! 权重加载层（架构文档 §4.1）。
//!
//! P2 目标：memmap2 零拷贝 + cudaMemcpyAsync 直达最终布局（FTW 格式）。
//! 当前先定义加载契约与配置解析。

use ft_core::{FtError, ModelConfig};
use std::path::{Path, PathBuf};

/// 模型目录探测结果。
#[derive(Debug, Clone)]
pub struct ModelDir {
    pub root: PathBuf,
    pub config: ModelConfig,
    pub safetensors: Vec<PathBuf>,
}

/// 从 HF 目录布局解析模型（读 config.json，枚举 *.safetensors）。
pub fn discover(root: &Path) -> Result<ModelDir, FtError> {
    let cfg_path = root.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| FtError::Invalid(format!("read {}: {e}", cfg_path.display())))?;
    let config: ModelConfig = serde_json_wrap(&raw)?;
    let mut safetensors = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let p = entry?.path();
        if p.extension().is_some_and(|e| e == "safetensors") {
            safetensors.push(p);
        }
    }
    safetensors.sort();
    Ok(ModelDir { root: root.to_path_buf(), config, safetensors })
}

// ft-core::ModelConfig 是 serde Serialize/Deserialize，这里包一层避免 ft-loader 直接依赖 serde_json 的公共 API 泄漏
fn serde_json_wrap<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, FtError> {
    serde_json::from_str(raw).map_err(|e| FtError::Invalid(format!("bad config.json: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_rejects_missing_config() {
        assert!(discover(Path::new("/nonexistent")).is_err());
    }
}
