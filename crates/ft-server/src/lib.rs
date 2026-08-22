//! API 服务层：OpenAI 兼容端点 + engine 线程桥。

pub mod api;
pub mod bridge;

pub use bridge::{spawn_engine, EngineHandle};
