//! 全层共享的 LRU 专家缓存。
//!
//! 对应 FreeToken 的 GPU expert cache（LRU · all layers）。
//! 句柄是 (layer, expert) 二元组，容量以 slot 计。

use std::collections::{HashMap, VecDeque};

/// 缓存键：(layer_id, expert_id)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

/// LRU 专家缓存。命中提升到队尾；未命中且满则淘汰队首。
pub struct LruExpertCache {
    cap: usize,
    map: HashMap<ExpertKey, ()>,
    order: VecDeque<ExpertKey>,
}

impl LruExpertCache {
    pub fn new(cap: usize) -> Self {
        Self { cap, map: HashMap::with_capacity(cap), order: VecDeque::with_capacity(cap) }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn contains(&self, k: ExpertKey) -> bool {
        self.map.contains_key(&k)
    }

    /// 查询是否命中；命中则提升。不插入。
    pub fn touch(&mut self, k: ExpertKey) -> bool {
        if !self.map.contains_key(&k) {
            return false;
        }
        if let Some(pos) = self.order.iter().position(|&x| x == k) {
            self.order.remove(pos);
            self.order.push_back(k);
        }
        true
    }

    /// 插入（未命中时调用）。满则淘汰最久未用。
    pub fn insert(&mut self, k: ExpertKey) {
        if self.map.contains_key(&k) {
            self.touch(k);
            return;
        }
        if self.map.len() >= self.cap && self.cap > 0 {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        if self.cap > 0 {
            self.map.insert(k, ());
            self.order.push_back(k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(l: u32, e: u32) -> ExpertKey {
        ExpertKey { layer: l, expert: e }
    }

    #[test]
    fn miss_then_hit_after_insert() {
        let mut c = LruExpertCache::new(2);
        assert!(!c.touch(k(0, 1)));
        c.insert(k(0, 1));
        assert!(c.touch(k(0, 1)));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn evicts_least_recent() {
        let mut c = LruExpertCache::new(2);
        c.insert(k(0, 1));
        c.insert(k(0, 2));
        c.touch(k(0, 1)); // 1 变热，2 变冷
        c.insert(k(0, 3)); // 应淘汰 2
        assert!(c.contains(k(0, 1)));
        assert!(!c.contains(k(0, 2)));
        assert!(c.contains(k(0, 3)));
    }

    #[test]
    fn zero_cap_never_hits() {
        let mut c = LruExpertCache::new(0);
        c.insert(k(0, 1));
        assert!(!c.touch(k(0, 1)));
        assert_eq!(c.len(), 0);
    }
}
