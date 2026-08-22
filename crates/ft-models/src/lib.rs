//! 模型定义层。Model trait 是 engine 与具体架构之间的边界。

use ft_core::{FtError, ModelConfig};

/// 模型执行契约（GPU 路径由 ft-kernel 承载）。
pub trait Model: Send {
    fn config(&self) -> &ModelConfig;
    /// prefill 一个 chunk，返回最后位置的 logits
    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>, FtError>;
    /// decode 一步（batch=1），返回 logits
    fn decode_step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, FtError>;
}

/// 已知架构注册表 —— 对齐 FreeToken docs/models.md 的 known-good 列表。
pub const KNOWN_ARCHS: &[&str] =
    &["DeepseekV4ForCausalLM", "Glm5MoeForCausalLM", "Qwen3MoeForCausalLM", "Gemma4ForCausalLM"];

pub fn is_known(arch: &str) -> bool {
    KNOWN_ARCHS.contains(&arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_archs() {
        assert!(is_known("DeepseekV4ForCausalLM"));
        assert!(!is_known("LlamaForCausalLM"));
    }
}
