//! 页式 KV pool。真实实现（DSA/SWA/radix）后续按 FreeToken kvcache/ 移植，
//! 本文件先给出 trait 契约 + 线性前缀匹配的朴素实现。

use ft_core::{FtError, SeqId};

/// 一个 KV 页能容纳的 token 数
pub const PAGE_TOKENS: usize = 16;

#[derive(Debug, Clone)]
pub struct SeqEntry {
    pub seq: SeqId,
    /// 已占用的页 id
    pub pages: Vec<u32>,
    pub num_tokens: usize,
}

/// KV 池契约：alloc / free / prefix 复用。
pub trait KvPool: Send {
    fn alloc(&mut self, seq: SeqId, num_tokens: usize) -> Result<SeqEntry, FtError>;
    fn extend(&mut self, entry: &mut SeqEntry, extra_tokens: usize) -> Result<(), FtError>;
    fn free(&mut self, entry: SeqEntry);
    /// 返回 (可复用的页数, 匹配 token 数)
    fn match_prefix(&self, tokens: &[u32]) -> Option<(u32, usize)>;
    fn stats(&self) -> KvStats;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvStats {
    pub total_pages: u32,
    pub used_pages: u32,
}

/// 朴素实现：单序列线性页分配 + 无 radix 复用（prefix match 仅同 prompt 完全一致时命中）。
pub struct SimpleKvPool {
    total_pages: u32,
    free_pages: Vec<u32>,
    // 记录 (prompt_hash, pages) 用于完全匹配复用
    registry: Vec<(u64, Vec<u32>)>,
}

impl SimpleKvPool {
    pub fn new(total_pages: u32) -> Self {
        Self {
            total_pages,
            free_pages: (0..total_pages).rev().collect(),
            registry: Vec::new(),
        }
    }

    fn take_pages(&mut self, n: usize) -> Result<Vec<u32>, FtError> {
        if n > self.free_pages.len() {
            return Err(FtError::Oom(format!(
                "kv pool: need {n} pages, {} free of {}",
                self.free_pages.len(),
                self.total_pages
            )));
        }
        Ok(self.free_pages.split_off(self.free_pages.len() - n))
    }
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    // FNV-1a
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl KvPool for SimpleKvPool {
    fn alloc(&mut self, seq: SeqId, num_tokens: usize) -> Result<SeqEntry, FtError> {
        let pages_needed = num_tokens.div_ceil(PAGE_TOKENS);
        let pages = self.take_pages(pages_needed)?;
        Ok(SeqEntry { seq, pages, num_tokens })
    }

    fn extend(&mut self, entry: &mut SeqEntry, extra_tokens: usize) -> Result<(), FtError> {
        let new_total = entry.num_tokens + extra_tokens;
        let pages_needed = new_total.div_ceil(PAGE_TOKENS);
        while entry.pages.len() < pages_needed {
            let p = self.take_pages(1)?;
            entry.pages.extend_from_slice(&p);
        }
        entry.num_tokens = new_total;
        Ok(())
    }

    fn free(&mut self, entry: SeqEntry) {
        self.free_pages.extend(entry.pages);
    }

    fn match_prefix(&self, tokens: &[u32]) -> Option<(u32, usize)> {
        let h = hash_tokens(tokens);
        self.registry.iter().find(|(rh, _)| *rh == h).map(|(_, pages)| (pages[0], tokens.len()))
    }

    fn stats(&self) -> KvStats {
        KvStats { total_pages: self.total_pages, used_pages: self.total_pages - self.free_pages.len() as u32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_extend_free_roundtrip() {
        let mut pool = SimpleKvPool::new(4);
        let mut e = pool.alloc(SeqId(1), 20).unwrap(); // 2 pages
        assert_eq!(e.pages.len(), 2);
        assert_eq!(pool.stats().used_pages, 2);
        pool.extend(&mut e, 30).unwrap(); // -> 50 tokens = 4 pages
        assert_eq!(e.pages.len(), 4);
        assert!(pool.alloc(SeqId(2), 1).is_err()); // 满
        pool.free(e);
        assert_eq!(pool.stats().used_pages, 0);
    }

    #[test]
    fn page_size_is_16() {
        assert_eq!(PAGE_TOKENS, 16);
    }
}
