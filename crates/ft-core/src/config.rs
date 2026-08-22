use serde::{Deserialize, Serialize};

/// MoE 层配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// MoE 层的 hidden（部分模型 expert 中间维不同）
    pub expert_hidden: usize,
    /// 参与路由的层索引；空 = 全部 decoder 层都是 MoE
    #[serde(default)]
    pub moe_layers: Vec<usize>,
}

/// 模型静态配置（对应 HF config.json 的子集）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub arch: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    #[serde(default)]
    pub moe: Option<MoeConfig>,
    /// 权重量化格式（专家权重）
    #[serde(default)]
    pub quant: Option<crate::dtype::Dtype>,
    /// 注意力类型：full / swa / dsa / hybrid
    #[serde(default)]
    pub attn_kind: String,
}

impl ModelConfig {
    pub fn moe_layers(&self) -> Vec<usize> {
        match &self.moe {
            None => vec![],
            Some(m) if m.moe_layers.is_empty() => (0..self.num_layers).collect(),
            Some(m) => m.moe_layers.clone(),
        }
    }
}
