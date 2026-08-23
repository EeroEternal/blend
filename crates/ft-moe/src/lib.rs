//! q* 放置策略（纯函数）。不再包含自研 MoE 执行器。

pub mod policy;

pub use policy::{BackendKind, BandwidthProfile, QStarPolicy};
