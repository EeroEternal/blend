//! 注意力后端 trait（对应 FreeToken attention/ 目录）。

/// 注意力类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    Full,
    /// sliding window
    Swa,
    /// DeepSeek sparse attention
    Dsa,
    /// 线性注意力 / recurrent state（GatedDeltaNet 等）
    LinearState,
}

pub trait AttentionBackend: Send + Sync {
    fn kind(&self) -> AttnKind;
    /// 该后端是否支持当前 GPU 架构（sm100 等）
    fn supports(&self, sm_major: u32) -> bool;
}

/// auto 选择逻辑的最小版本：按模型 attn_kind + 架构能力挑后端。
pub struct BackendEntry {
    pub name: &'static str,
    pub kind: AttnKind,
}

pub const KNOWN_BACKENDS: &[BackendEntry] = &[
    BackendEntry { name: "fa", kind: AttnKind::Full },
    BackendEntry { name: "trtllm", kind: AttnKind::Full },
    BackendEntry { name: "dsv4_sparse", kind: AttnKind::Dsa },
    BackendEntry { name: "dsa", kind: AttnKind::Dsa },
];

pub fn auto_pick(kind: AttnKind) -> &'static str {
    match KNOWN_BACKENDS.iter().find(|b| b.kind == kind) {
        Some(b) => b.name,
        None => "fa",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_pick_dsa() {
        assert_eq!(auto_pick(AttnKind::Dsa), "dsv4_sparse");
    }

    #[test]
    fn fallback_fa() {
        assert_eq!(auto_pick(AttnKind::LinearState), "fa");
    }
}
