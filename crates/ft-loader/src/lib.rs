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
/// 从 model.safetensors.index.json 的 weight_map 解析张量所在分片。
pub fn locate_tensor(root: &Path, name: &str) -> Result<PathBuf, FtError> {
    let idx = root.join("model.safetensors.index.json");
    let raw = std::fs::read_to_string(&idx)
        .map_err(|e| FtError::Invalid(format!("read {}: {e}", idx.display())))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| FtError::Invalid(format!("index json: {e}")))?;
    let shard = v["weight_map"][name]
        .as_str()
        .ok_or_else(|| FtError::Invalid(format!("tensor not in index: {name}")))?;
    Ok(root.join(shard))
}

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

// ─── safetensors 读取（memmap 零拷贝）─────────────────────────────

/// safetensors 文件：8 字节 LE header 长度 + JSON header + 对齐的数据区。
pub struct SafeTensorFile {
    mmap: memmap2::Mmap,
    data_offset: usize,
    index: std::collections::BTreeMap<String, TensorMeta>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

impl SafeTensorFile {
    pub fn open(path: &Path) -> Result<Self, FtError> {
        let f = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        if mmap.len() < 8 {
            return Err(FtError::Invalid(format!("{}: too small", path.display())));
        }
        let hdr_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let end = 8 + hdr_len;
        if end > mmap.len() {
            return Err(FtError::Invalid(format!("{}: bad header length", path.display())));
        }
        let hdr: std::collections::BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..end])
                .map_err(|e| FtError::Invalid(format!("{}: bad header json: {e}", path.display())))?;
        let mut index = std::collections::BTreeMap::new();
        for (name, v) in hdr {
            if name == "__metadata__" {
                continue;
            }
            let meta: TensorMeta = serde_json::from_value(v)
                .map_err(|e| FtError::Invalid(format!("{name}: {e}")))?;
            index.insert(name, meta);
        }
        Ok(Self { mmap, data_offset: end, index })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    pub fn meta(&self, name: &str) -> Option<&TensorMeta> {
        self.index.get(name)
    }

    /// 张量视图：零拷贝，生命周期绑定文件 mmap。
    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let m = self.index.get(name)?;
        let (s, e) = m.data_offsets;
        Some(TensorView {
            dtype: m.dtype.as_str(),
            shape: &m.shape,
            data: &self.mmap[self.data_offset + s..self.data_offset + e],
        })
    }
}

pub struct TensorView<'a> {
    pub dtype: &'a str,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// F32 小端读取；其他 dtype 显式报错（bf16/fp4 解码在 kernel 层做）。
    pub fn as_f32(&self) -> Result<Vec<f32>, FtError> {
        if self.dtype != "F32" {
            return Err(FtError::Invalid(format!("dtype {} not supported as f32", self.dtype)));
        }
        Ok(self.data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// BF16 原始位；F16 同样按 2 字节取出。
    pub fn as_u16(&self) -> Result<Vec<u16>, FtError> {
        if self.dtype != "BF16" && self.dtype != "F16" {
            return Err(FtError::Invalid(format!("dtype {} not u16", self.dtype)));
        }
        Ok(self.data.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// BF16 → f32（左移 16 位）。
    pub fn as_f32_from_bf16(&self) -> Result<Vec<f32>, FtError> {
        let bits = self.as_u16()?;
        Ok(bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect())
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手工构造一个最小 safetensors 文件（F32 [2,3]）。
    fn write_fixture(path: &Path) {
        let vals: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data = Vec::new();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let header = format!(
            r#"{{"w":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{}]}}}}"#,
            data.len()
        );
        // safetensors 要求 header 8 字节对齐（pad 空格）
        let pad = (8 - (header.len() % 8)) % 8;
        let mut out = Vec::new();
        out.extend_from_slice(&((header.len() + pad) as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend(std::iter::repeat(b' ').take(pad));
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn parse_and_read_f32() {
        let p = std::env::temp_dir().join("blend-st.st");
        write_fixture(&p);
        let st = SafeTensorFile::open(&p).unwrap();
        assert_eq!(st.names().cloned().collect::<Vec<_>>(), vec!["w"]);
        let tv = st.tensor("w").unwrap();
        assert_eq!(tv.shape, &[2, 3]);
        assert_eq!(tv.numel(), 6);
        assert_eq!(tv.as_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(st.tensor("missing").is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn discover_rejects_missing_config() {
        assert!(discover(Path::new("/nonexistent")).is_err());
    }
}
