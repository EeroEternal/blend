//! 引擎核心：调度器状态机 + engine 线程消息协议。
//!
//! 设计约束（架构文档 §1）：
//! - engine 运行在普通 OS 线程的事件循环里，不碰 async
//! - 与 server 之间只通过 flume channel 传消息

pub mod scheduler;
pub mod step;

pub use scheduler::{Scheduler, SchedulerConfig};
pub use step::Step;
