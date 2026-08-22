//! 每步 MoE 执行计划：显式化 "哪些 miss 走 PCIe、哪些 CPU 就地算"，
//! 让 engine 层可以记录、限流、覆盖（fork 后可控性的落点）。

use crate::policy::QStarPolicy;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissAction {
    /// 经 PCIe 取到 GPU expert cache
    Fetch { expert_id: u32 },
    /// 留在 CPU 内存就地计算
    CpuCompute { expert_id: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MoePlan {
    pub cache_hits: Vec<u32>,
    pub misses: Vec<MissAction>,
}

impl MoePlan {
    /// 由路由结果 + 缓存命中情况生成计划。
    ///
    /// * `routed`      — 本步所有被激活的 (layer 内) 专家 id
    /// * `is_cached`   — 该专家是否在 GPU LRU 缓存中
    pub fn build(routed: &[u32], is_cached: impl Fn(u32) -> bool, q: &QStarPolicy) -> Self {
        let mut cache_hits = Vec::new();
        let mut miss_ids = Vec::new();
        for &e in routed {
            if is_cached(e) {
                cache_hits.push(e);
            } else {
                miss_ids.push(e);
            }
        }
        let n_fetch = q.fetch_count(miss_ids.len());
        // 取回优先选择"更热"的专家——朴素策略下按 id 序即可，真实 LRU 热度由 cache 层给出
        let mut misses = Vec::with_capacity(miss_ids.len());
        for (i, e) in miss_ids.iter().enumerate() {
            let action = if i < n_fetch {
                MissAction::Fetch { expert_id: *e }
            } else {
                MissAction::CpuCompute { expert_id: *e }
            };
            misses.push(action);
        }
        Self { cache_hits, misses }
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.cache_hits.len(),
            self.misses.iter().filter(|m| matches!(m, MissAction::Fetch { .. })).count(),
            self.misses.iter().filter(|m| matches!(m, MissAction::CpuCompute { .. })).count(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::BandwidthProfile;

    #[test]
    fn plan_splits_misses_by_qstar() {
        let q = QStarPolicy::calibrate(&BandwidthProfile::pro6000_epyc9355());
        let routed: Vec<u32> = (0..100).collect();
        let cached = |e: u32| e < 88; // 88 hits, 12 misses
        let plan = MoePlan::build(&routed, cached, &q);
        let (hits, fetch, cpu) = plan.stats();
        assert_eq!(hits, 88);
        assert_eq!(fetch + cpu, 12);
        assert_eq!(fetch, q.fetch_count(12));
        assert_eq!(cpu, 12 - fetch);
    }

    #[test]
    fn all_hit_no_miss() {
        let q = QStarPolicy::calibrate(&BandwidthProfile::pro6000_epyc9355());
        let plan = MoePlan::build(&[1, 2, 3], |_| true, &q);
        assert_eq!(plan.stats(), (3, 0, 0));
    }
}
