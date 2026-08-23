//! Hybrid 运行时：LRU 缓存 + q\* 决策，不含具体计算内核。
//!
//! 计算由调用方按 `MoePlan` 执行（CPU SIMD / GPU FFI），本模块只负责：
//! 路由 → 计划 → 更新缓存 → 累计统计。

use crate::cache::{ExpertKey, LruExpertCache};
use crate::plan::{MissAction, MoePlan};
use crate::policy::QStarPolicy;

#[derive(Debug, Default, Clone)]
pub struct HybridStats {
    pub hits: u64,
    pub fetches: u64,
    pub cpu_misses: u64,
}

impl HybridStats {
    pub fn total(&self) -> u64 {
        self.hits + self.fetches + self.cpu_misses
    }
    pub fn hit_rate(&self) -> f64 {
        let t = self.total();
        if t == 0 { 0.0 } else { self.hits as f64 / t as f64 }
    }
}

/// 一层一次 decode 的 hybrid 决策器。
pub struct HybridRuntime {
    cache: LruExpertCache,
    q: QStarPolicy,
    stats: HybridStats,
}

impl HybridRuntime {
    pub fn new(cache_slots: usize, q: QStarPolicy) -> Self {
        Self { cache: LruExpertCache::new(cache_slots), q, stats: HybridStats::default() }
    }

    pub fn stats(&self) -> &HybridStats {
        &self.stats
    }
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// 为本层路由出计划，并更新 LRU / 统计。
    pub fn plan_layer(&mut self, layer: u32, routed: &[u32]) -> MoePlan {
        let cache = &self.cache;
        let plan = MoePlan::build(routed, |e| cache.contains(ExpertKey { layer, expert: e }), &self.q);
        let (h, f, c) = plan.stats();
        self.stats.hits += h as u64;
        self.stats.fetches += f as u64;
        self.stats.cpu_misses += c as u64;

        for &e in &plan.cache_hits {
            self.cache.touch(ExpertKey { layer, expert: e });
        }
        for m in &plan.misses {
            if let MissAction::Fetch { expert_id } = m {
                self.cache.insert(ExpertKey { layer, expert: *expert_id });
            }
        }
        plan
    }

    /// 把 CpuCompute 专家写成 ids 数组（其余 -1），供内核跳过。
    pub fn cpu_ids(plan: &MoePlan, k: usize) -> Vec<i32> {
        let mut ids = vec![-1i32; k];
        let mut i = 0;
        for m in &plan.misses {
            if let MissAction::CpuCompute { expert_id } = m {
                if i < k {
                    ids[i] = *expert_id as i32;
                    i += 1;
                }
            }
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::BandwidthProfile;

    #[test]
    fn warmup_then_hits_with_sticky_routing() {
        let q = QStarPolicy::calibrate(&BandwidthProfile::pro6000_epyc9355());
        let mut rt = HybridRuntime::new(64, q);
        let sticky = [1u32, 2, 3, 4, 5];
        // 多步：每次 5 个热专家 + 1 个新专家（放末位，让 Fetch 优先填热集）
        for step in 0..12 {
            let mut routed = sticky.to_vec();
            routed.push(10 + step);
            rt.plan_layer(0, &routed);
        }
        // 热集应已在缓存，后续命中率显著上升
        let before = rt.stats().hits;
        for step in 12..20 {
            let mut routed = sticky.to_vec();
            routed.push(30 + step);
            rt.plan_layer(0, &routed);
        }
        let after = rt.stats().hits - before;
        // 8 步 × 5 热专家，至少命中大半
        assert!(after >= 8 * 3, "expected sticky hits, got {after}");
    }
}
