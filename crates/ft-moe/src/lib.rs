//! MoE 执行子系统：带宽自适应 q\* 策略（纯函数）、执行计划、CPU 执行器。

pub mod cpu;
pub mod plan;
pub mod policy;
pub mod topology;

pub use cpu::{CpuMoeExecutor, NaiveF32Executor};
pub use plan::{MoePlan, MissAction};
pub use policy::{BackendKind, BandwidthProfile, QStarPolicy};
pub use topology::physical_cores;
