/// 代数标记句柄。free 后复用槽位会递增 gen，旧句柄自动失效。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle {
    pub index: u32,
    pub gen: u32,
}

/// 泛型槽位表：O(1) alloc/free，代数防 use-after-free。
/// gen 独立存储，不随 T 释放而丢失。
#[derive(Debug)]
pub struct SlotTable<T> {
    entries: Vec<Option<T>>,
    gens: Vec<u32>,
    free: Vec<u32>,
    live: usize,
}

impl<T> SlotTable<T> {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: (0..cap).map(|_| None).collect(),
            gens: vec![0; cap],
            free: (0..cap as u32).rev().collect(),
            live: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.live
    }
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }
    /// 剩余可分配槽数
    pub fn remaining(&self) -> usize {
        self.free.len()
    }

    pub fn insert(&mut self, v: T) -> Result<Handle, ft_core::FtError> {
        let idx = self.free.pop().ok_or_else(|| {
            ft_core::FtError::Oom(format!("slot table full ({})", self.entries.len()))
        })?;
        let i = idx as usize;
        self.gens[i] = self.gens[i].wrapping_add(1);
        let gen = self.gens[i];
        self.entries[i] = Some(v);
        self.live += 1;
        Ok(Handle { index: idx, gen })
    }

    fn check(&self, h: Handle) -> Result<(), ft_core::FtError> {
        match h.index as usize {
            i if i < self.entries.len() => {
                if self.entries[i].is_none() || self.gens[i] != h.gen {
                    Err(ft_core::FtError::StaleHandle { handle: h.gen, current: self.gens[i] })
                } else {
                    Ok(())
                }
            }
            _ => Err(ft_core::FtError::StaleHandle { handle: h.gen, current: 0 }),
        }
    }

    pub fn get(&self, h: Handle) -> Result<&T, ft_core::FtError> {
        self.check(h)?;
        Ok(self.entries[h.index as usize].as_ref().unwrap())
    }

    pub fn get_mut(&mut self, h: Handle) -> Result<&mut T, ft_core::FtError> {
        self.check(h)?;
        Ok(self.entries[h.index as usize].as_mut().unwrap())
    }

    pub fn remove(&mut self, h: Handle) -> Result<T, ft_core::FtError> {
        // gen 不清零：复用槽位时 insert 会递增，旧句柄永远失效
        self.check(h)?;
        let v = self.entries[h.index as usize].take().unwrap();
        self.live -= 1;
        self.free.push(h.index);
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut t: SlotTable<String> = SlotTable::with_capacity(2);
        let h1 = t.insert("a".into()).unwrap();
        let h2 = t.insert("b".into()).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t.get(h1).unwrap(), "a");
        assert_eq!(t.remove(h2).unwrap(), "b");
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn stale_handle_detected_after_reuse() {
        let mut t: SlotTable<u64> = SlotTable::with_capacity(1);
        let h1 = t.insert(42).unwrap();
        assert_eq!(t.remove(h1).unwrap(), 42);
        // 复用同一物理槽位，gen 递增
        let h2 = t.insert(7).unwrap();
        assert_eq!(h2.index, h1.index);
        assert_ne!(h2.gen, h1.gen);
        // 旧句柄访问必须报 StaleHandle 而不是读到 7
        assert!(matches!(t.get(h1), Err(ft_core::FtError::StaleHandle { .. })));
        assert!(matches!(t.get_mut(h1), Err(ft_core::FtError::StaleHandle { .. })));
        assert_eq!(*t.get(h2).unwrap(), 7);
    }

    #[test]
    fn oom_when_full() {
        let mut t: SlotTable<u8> = SlotTable::with_capacity(1);
        t.insert(1).unwrap();
        assert!(matches!(t.insert(2), Err(ft_core::FtError::Oom(_))));
    }
}
