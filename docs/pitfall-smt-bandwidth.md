# 坑记录：SMT 满载使内存带宽倒亏 27%

> 日期：2026-08-22
> 环境：EPYC 9355，755GB DDR5，blend mem-bench 实测
> 影响：所有 CPU 侧批量计算（MoE 执行器、权重搬运、量化/反量化）的线程池配置

---

## 现象

`blend mem-bench`（顺序读内存求和，STREAM-like）在同一台机器上的结果：

| 配置 | 读带宽 | 相对损失 |
|---|---|---|
| 1 线程 | 43.7 GB/s | — |
| **64 线程（= 物理核数）** | **225.7 GB/s** | 基准 |
| 128 线程（SMT 满载） | 162.7 GB/s | **−27%** |

## 根因

内存带宽受限任务在物理核数处即达 DRAM 饱和；继续开 SMT 兄弟线程只增加干扰：

1. 同核两个流式读指针互踩 L1/L2 cache line
2. TLB 压力翻倍
3. OS 调度漂移可能把两个线程排到同一物理核而其他核空闲

SMT 的收益场景是计算密集 + 流水线气泡（分支误预测、依赖等待）；纯流式读没有气泡可填，兄弟线程只有成本没有收益。

## FreeToken 的对应设计（真机日志佐证）

启动 DeepSeek-V4-Flash 时：

```
CPU MoE executor ready: threads=63 (pinned to cores 0..62)
torch intra-op threads: 64 -> 1 (cores reserved for the pinned CPU MoE pool)
```

两条措施精确对应两个坑：
- 线程数 = 物理核数 − 1，不用逻辑核数
- core pinning 固定亲和性，主进程 intra-op 线程降到 1，防止运行期漂移与缓存污染

## blend 的落地决策

- [ ] rayon / 手写线程池按**物理核数**配置：用 `sysinfo` 或解析
  `/proc/cpuinfo` 的 `core id`（`available_parallelism()` 返回逻辑核数，不能直接用）
- [ ] CPU MoE 执行器工作线程 pin 到固定物理核集合（候选 crate：`core_affinity`）
- [ ] engine 主线程与 server tokio runtime 不与 MoE 工作核重叠
- [ ] mem-bench 增加 `--threads` 自动探测物理核数的默认值
- [ ] NUMA 注意项（后续）：多路服务器上优先本地节点分配（first-touch），跨 socket 访问会再打折

## 复现命令

```bash
./target/release/blend mem-bench --gib 8 --iters 3 --threads 64   # ≈225 GB/s
./target/release/blend mem-bench --gib 8 --iters 3 --threads 128  # ≈163 GB/s（更慢）
```
