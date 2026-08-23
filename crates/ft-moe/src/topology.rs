//! CPU 拓扑检测。
//!
//! 坑记录（docs/pitfall-smt-bandwidth.md）：线程池必须按物理核数配置，
//! SMT 满载会使内存带宽倒亏 ~27%。`available_parallelism()` 返回逻辑核数，
//! 不能直接用。

/// 物理核数。Linux 解析 /proc/cpuinfo 的唯一 (physical id, core id) 对；
/// 其他平台退化为 available_parallelism()（并注明可能含 SMT）。
pub fn physical_cores() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Some(n) = parse_proc_cpuinfo() {
            return n;
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

#[cfg(target_os = "linux")]
fn parse_proc_cpuinfo() -> Option<usize> {
    let raw = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut pairs = std::collections::BTreeSet::new();
    let mut phys: Option<u32> = None;
    let mut core: Option<u32> = None;
    for line in raw.lines() {
        if let Some(v) = line.strip_prefix("physical id") {
            phys = v.trim_start_matches(|c: char| c == ':' || c.is_whitespace())
                .parse::<u32>()
                .ok();
        } else if let Some(v) = line.strip_prefix("core id") {
            core = v.trim_start_matches(|c: char| c == ':' || c.is_whitespace())
                .parse::<u32>()
                .ok();
        } else if line.starts_with("processor") {
            // 新 processor 段开始：flush 上一对
            if let (Some(p), Some(c)) = (phys.take(), core.take()) {
                pairs.insert((p, c));
            }
        }
    }
    if let (Some(p), Some(c)) = (phys, core) {
        pairs.insert((p, c));
    }
    if pairs.is_empty() { None } else { Some(pairs.len()) }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn parses_at_least_one_and_not_more_than_logical() {
        let phys = physical_cores();
        let logical = std::thread::available_parallelism().unwrap().get();
        assert!(phys >= 1);
        assert!(phys <= logical, "physical {phys} must be <= logical {logical}");
    }
}
