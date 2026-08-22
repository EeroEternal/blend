/// 序列全局唯一 id。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(pub u64);

/// 进入引擎的单请求运行态。
#[derive(Debug, Clone)]
pub struct ReqState {
    pub seq: SeqId,
    /// 已被调度确认的 prompt token
    pub prompt_tokens: Vec<u32>,
    /// KV pool 中的序列句柄（由 ft-memory 分配）
    pub kv_handle: Option<u64>,
    pub generated: Vec<u32>,
    pub max_new_tokens: usize,
}
