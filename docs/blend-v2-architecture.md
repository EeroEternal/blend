# blend v2：单机推理服务框架

> 日期：2026-08-23  
> 取代：[freetoken-rust-architecture.md](./freetoken-rust-architecture.md) 里「Rust 重写整网 forward」的路线  
> 前提：实测同卡 Qwen3-30B，FreeToken（PyTorch forward）130 tok/s，blend 自研层循环 ~8 tok/s。差距在执行模型，不在语言。

---

## 0. 产品定位

blend 是 **单机推理服务框架**，不是又一个从零写的 LLM kernel 仓库。

要解决的问题：

1. 一台机器上的 **1 / 2 / 4 / 8 卡 + 大内存 CPU**，把模型和请求放对地方  
2. 对上接 **Agent**（多轮、工具、前缀缓存、多模型路由），而不是只做一个 `/v1/chat/completions`  
3. 对下复用已经打赢过的 GPU forward（PyTorch + FlashInfer + fused MoE），不自己再焊 48 层 hidden

一句话：**Rust 管「谁在哪张卡、用多少 CPU、怎么服务 Agent」；PyTorch worker 管「这一步 token 怎么算」。**

---

## 1. 为什么改路线

原设计（v1）把引擎核心也 Rust 化：自己管 KV、自己 launch 每层 kernel。结果是：

- 单算子能对拍、q\* / FI / hybrid 机制正确  
- 整网 decode 每层 D2H，录不成 CUDA Graph  
- 同模型慢 ~15×

PyTorch 白送的是「`hidden` 一直是 GPU Tensor + `model.forward()` 可被 Graph 录下来」。这不是 Rust 做不到，是我们没接这根水管。v2 把这根水管接回来，Rust 只做它更擅长的编排。

---

## 2. 总体结构

```
                    Agent / IDE / 网关
                           │
              OpenAI  /  Anthropic  / 内部 gRPC
                           │
┌──────────────────────────▼──────────────────────────┐
│  blend-control  (Rust, tokio)                       │
│  路由 · 会话 · 前缀缓存索引 · 配额 · 工具协议           │
│  放置策略 (Placement): 模型 → 设备组 + 并行度 + q*    │
└─────────────┬────────────────────┬──────────────────┘
              │ shm / UDS / NCCL   │
              ▼                    ▼
┌─────────────────────┐  ┌─────────────────────────────┐
│ torch-worker × N    │  │ cpu-pool (可选)              │
│ Python + PyTorch    │  │ 主机专家 / 我们的 AVX512     │
│ = FreeToken 或      │  │ 由 worker 调，不由 control   │
│   薄封装的 HF+FI    │  │ 直接编层循环                 │
│ 整网 forward + Graph│  └─────────────────────────────┘
│ TP=1/2/4 在组内 NCCL│
└─────────────────────┘
              │
     GPU0..GPU7 中的一个子集
```

进程模型：

- **1 个 control 进程**（无 GPU，或只做 IPC）  
- **N 个 worker 进程**，每个绑定一组 GPU（`CUDA_VISIBLE_DEVICES`）  
- 同模型多副本 = 多个 worker；单模型切 2/4 卡 = 一个 worker 内 TP

不要把 8 卡塞进一个 Python 进程再自己做调度——卡间隔离、崩溃、升级都以进程为单位。

---

## 3. 放置策略（2 卡 / 4 卡 / CPU 的核心）

把原来的 q\* 从「一层里 miss 怎么拆」升级成 **整模型放哪**。

### 3.1 三种放置

| 模式 | 适用 | 做法 |
|---|---|---|
| `replica` | 小模型、高 QPS、Agent 多会话 | 每卡一个完整副本，control 做请求级负载均衡 |
| `tp` | 单模型放不进 1 卡（权重大 / KV 大） | 2/4 卡张量并行，一个 worker，NCCL |
| `hybrid` | 专家池 ≫ 单卡显存（DSV4 / GLM-753B） | 1 卡（或 TP 组）+ 主机专家库 + GPU LRU，**q\* 仍有效** |

一台 8 卡机上可以同时存在：

```
GPU0-1  tp=2   GLM-5.2 NVFP4 hybrid（主机 420GB 专家）
GPU2-3  replica×2  Qwen3-30B（Agent 主模型）
GPU4    replica    小模型 / speculator
GPU5-7  空或第三个模型
CPU     专家 offload + tokenizer + control
```

### 3.2 决策输入（沿用已有标定）

`ft-bench` / `mem-bench` / `pcie-bench` 已经能量：

- `B_cpu`、`B_pcie`、GPU 名、空闲显存  
- q\*：`fetch_frac ≈ B_pcie / (B_pcie + B_cpu_overlap)`  

v2 增加：

- 模型：参数量、激活量、是否 MoE、推荐并行度  
- 机器：拓扑（PCIe 交换机、NUMA）、每卡空闲显存  
- 负载：Agent 会话数、前缀命中率、TTFT vs TPOT 权重  

输出一份 `PlacementPlan`（纯数据，可单测，和现在的 `QStarPolicy` 同一风格）：

```text
model=qwen3-30b  mode=replica  gpus=[2,3]  dtype=bf16  moe=offload
model=glm-5.2    mode=hybrid   gpus=[0,1]  tp=2       moe=hybrid  q_star=0.19
```

### 3.3 2 卡 / 4 卡具体怎么切

- **2 卡同一模型**：优先 TP=2（KV 减半、权重切列）。两卡 PCIe 对打走 NCCL，worker 内完成，control 只看见一个 endpoint。  
- **4 卡**：30B 级一般 **replica×4** 比 TP=4 更合适（Agent 要的是并发会话，不是单条更长）。70B+/大 KV 再用 TP=2 × replica×2。  
- **不要默认 TP=8**：单机 8 卡全绑一个模型，Agent 多会话时尾延迟更差。  
- **CPU**：只承担 (1) tokenizer / 调度 (2) MoE 专家池与 q\* miss (3) 绝不把 attention 放回 CPU（已实测是 15× 的主因）。

---

## 4. Worker：PyTorch 整网 forward

worker 不从零写。两种实现，control 只认同一套 RPC：

| Worker | 何时用 |
|---|---|
| **FreeToken 进程**（首选） | 已支持的 MoE（Qwen3 / DSV4 / GLM），要 130 tok/s 这一档 |
| **blend-torch**（薄封装） | FT 还不支持的结构；HF + FlashInfer + 可选 torch.compile / CUDA Graph |

RPC（gRPC 或 UDS + protobuf，不要 HTTP 进热路径）：

```
Load(model, placement) → ready
Prefill(session, tokens, prefix_id?) → kv_handle, logits?
Decode(session, token | sampling) → token_stream
Unload / ResizeCache / Health
```

约束（从这次踩坑写死）：

- worker 内 **禁止** 把 `hidden` 拷回 control  
- decode 热路径只收采样参数，吐 token id  
- CUDA Graph 的捕获、replay 全在 worker；control 不知道 Graph 存在  

我们已有的 AVX512 / `libftkernels` **可以**作为 worker 的 CPU 后端插件（q\* miss），不要再从 Rust control 调它们编层。

---

## 5. 对上：Agent

Agent 不是「多打几个 HTTP」。要当一等公民：

| 能力 | 做法 |
|---|---|
| 协议 | 继续 OpenAI + Anthropic；另加 **内部 gRPC**（工具结果回灌、取消、会话亲和） |
| 会话亲和 | 同一 `session_id` 钉在同一 replica，前缀 KV / radix 才命中 |
| 前缀缓存 | worker 内 radix（FT 已有）；control 只存「哪台 worker 有这段 prefix」的索引 |
| 工具轮 | 语义锚点（`<think>` / `tool_call`）复用 FT 的 checkpoint 思想，避免整段 re-prefill |
| 多模型 | Agent 的 planner 用小模型、actor 用 30B：control 按 `model` 字段路由，不要一个 worker 切模型 |
| 启动器 | 保留 `blend launch claude/codex/...`，只改 endpoint 指向 control，不指向单个 FT |

control 需要的状态（全在 Rust，可持久化）：

```
Session { id, worker, prefix_hash, model, tool_epoch }
Worker { id, gpus, models[], inflight, kv_used }
```

不要把对话历史当「每次整包 prompt」——那是 TTFT 爆炸的根因。

---

## 6. 和现有 crate 怎么衔接

不推倒重来，改职责：

| 现有 | v2 角色 |
|---|---|
| `ft-server` / `ft-cli` | 长成 **control**：多 worker 路由、放置、Agent API |
| `ft-moe::policy` / `ft-bench` / `pcie-bench` | **放置与 q\* 的纯函数库**，继续用，单测保留 |
| `ft-engine` DecodeDriver / HybridRuntime | 降级为「若自研 worker 才用」；默认路径不再走它编 48 层 |
| `decode-qwen` / 自研 GEMV | 保留作 **对拍与教学路径**，不作为生产 decode |
| `libftkernels` / AVX512 shim | 可链进 **torch worker 的 CPU 插件**，不从 control 调 |
| FreeToken 部署经验 | worker 的默认实现就是 `ft serve` |

生产路径一句话：

```
Agent → blend-control → (UDS) → ft serve | blend-torch
```

---

## 7. 分期

| 期 | 交付 | 验收（这台 8×6000） |
|---|---|---|
| **V2.0** | control 反代单个 FT worker；OpenAI/Anthropic 不变 | Qwen3 经 blend 打到 **≥100 tok/s**（贴近 FT 130） |
| **V2.1** | 放置：replica ×2 / ×4；会话粘滞 | 2 个 Qwen3 副本，并发会话 QPS 近似线性 |
| **V2.2** | TP=2 worker；GLM / DSV4 hybrid 仍走 FT | 2 卡 TP 能拉起更大模型或更长 KV |
| **V2.3** | Agent：前缀索引 + `launch` + 工具轮不重 prefill | 真实 Claude Code / Codex 钉会话 |
| **V2.4** | 可选：自研 worker 的「整层不 D2H」；AVX512 作 FT 的 CPU 插件 | 不阻塞上面四期 |

V2.0 若做不到 100 tok/s，说明反代或协议把 Graph 热路径打断了，先修这个，不要回头焊层。

---

## 8. 明确不做

- 用 Rust 再实现一遍 `Qwen3MoeForCausalLM.forward`  
- 默认把 attention 放 CPU  
- control 进程碰 CUDA  
- 单进程占满 8 卡再在进程内「灵活调度」  
- 为了多卡先上复杂 PP（流水并行）；单机 2/4 卡 TP + replica 够用  

---

## 9. 一句话

v1 把 Rust 用在 forward 上，输掉了 PyTorch 白送的设备图。  
v2 把 Rust 用在 **框架** 上：Agent、多模型、2/4 卡放置、CPU 专家与 GPU 的分工（q\*）。  
算力层承认现实——**整网 GPU forward 用 PyTorch/FreeToken**。
