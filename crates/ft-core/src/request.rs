use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SamplingParams {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub sampling: SamplingParams,
    /// OpenAI 顶层的 max_tokens（与 sampling.max_tokens 取先到者）
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

impl ChatRequest {
    pub fn effective_max_tokens(&self, default_cap: usize) -> usize {
        self.sampling
            .max_tokens
            .or(self.max_tokens)
            .unwrap_or(default_cap)
    }
}
