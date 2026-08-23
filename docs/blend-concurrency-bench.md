# 并发怎么考

单流 tok/s（FreeToken 日志里的 `gen throughput`）**不能**当并发指标。那是 GPU 在一条 decode 上的瞬时速度。考核框架要看：**N 路同时打进来时，总量和尾延迟怎么变。**

## 1. 该报的数

| 指标 | 含义 | 怎么算 |
|---|---|---|
| **N** | 同时在飞的请求数 | 固定并发，发齐再等齐 |
| **agg tok/s** | 系统总输出 | 全部 `completion_tokens` / 墙钟（第一发→最后收） |
| **p50 / p99 时延** | 用户体验 | 每个请求从发到收齐的 ms |
| **ok/fail** | 过载是否丢请求 | HTTP 非 2xx |
| （可选）TTFT p99 | Agent 是否觉得卡 | 第一个 token 的时间 |

不要用「N 路各自的 decode tok/s 再平均」——会把排队藏掉。

## 2. 和放置的关系（必须分开考）

| 拓扑 | 期望 |
|---|---|
| **1 replica** | N 增大：agg 先升后平台，p99 变差（同卡 batch） |
| **R 个 replica** | N ≤ R 时 agg ≈ 线性；N > R 后像「每卡上再堆 N/R」 |
| **TP=2（一份权重）** | 只有 **1 个**逻辑 worker。N 增大是 **同模型 batch**，不是 2 倍副本。agg 不会按卡数翻倍 |

把 TP=2 和 replica×2 放在一张「并发图」里比，是常见误考。

## 3. 命令

```bash
# 打 control（后面可以是 1 副本 / 2 副本 / 1 个 TP2）
blend bench-conc \
  --url http://127.0.0.1:8080 \
  --model Qwen3-30B-A3B-Instruct \
  --concurrency 1,2,4,8,16 \
  --max-tokens 128
```

每个请求带不同 `x-blend-session`，才会拆到不同 replica。若要压 **单个** TP worker，把 `--url` 指到 `:1940`。

## 4. 2026-08-23 本机扫了一轮（control 后挂 2 replica + 1×TP2）

| N | agg tok/s | p50 | p99 | 解读 |
|---|---|---|---|---|
| 1 | 50 | 1.9s | 1.9s | **含 prefill 的端到端**，低于日志里的 140 decode-only |
| 2 | 100 | 1.9s | 1.9s | ≈2×，两路落到两张卡 |
| 4 | 97 | 3.9s | 3.9s | 3 个 worker 上挤 4 路，时延翻倍、总量没涨 |
| 8 | 183 | 2.9s | 4.2s | 总量再上去，p99 变差（过载征兆） |

结论写法：

- 副本数决定「N 较小时能不能线性」  
- 单卡 batch 决定「N 超过副本后还能不能再涨总量」  
- **合格**：指定 SLO（例如 p99 < 3s）下的最大 agg tok/s，而不是无约束的峰值

## 5. Agent 场景还要加一维

真实 Agent 是长会话 + 前缀命中，不是 N 条互不相干的短请求。补考：

1. 同一 `x-blend-session` 连打 10 轮（应粘在同一 replica，TTFT 应变短）  
2. 故意打到错误 replica（应看到 TTFT 回升）  

那是 V2.3 前缀索引的验收，不是这张并发表能代替的。
