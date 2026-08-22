//! 内存子系统：generation-based slot 表、页式 KV pool。
//!
//! 设计约束（对应架构文档 §4.1）：
//! - 所有句柄带 generation，stale handle 是编译期无法表达但运行期必须抓住的错误
//! - unsafe 只允许出现在 pinned buffer / DMA 路径，本 crate 保持 100% safe

pub mod kv;
pub mod slot;

pub use kv::{KvPool, SimpleKvPool};
pub use slot::{Handle, SlotTable};
