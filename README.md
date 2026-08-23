# blend

单机推理**调度**服务。算力层是 [FreeToken](https://github.com/FlashML-org/FreeToken)（PyTorch + Triton / FlashInfer + CUDA Graph）。

```
Agent / curl  →  blend control  →  ft serve [--tp-size 2]
```

## 命令

```bash
# 拉起 FreeToken worker
blend spawn --model ~/models/Qwen3-30B-A3B-Instruct --gpus 0,1 --tp 2 --port 1940

# 网关（可挂多个 replica / 一个 TP 组）
blend control --port 8080 \
  --worker http://127.0.0.1:1930 \
  --worker 'http://127.0.0.1:1940#tp=2,label=qwen-tp2'

# 同 session TTFT / 并发扫
blend bench-session --url http://127.0.0.1:8080 --turns 6
blend bench-conc --url http://127.0.0.1:8080 --concurrency 1,2,4,8
```

会话头 `x-blend-session` 钉 worker；详见 [docs/blend-v2-architecture.md](docs/blend-v2-architecture.md)。

## 仓库

| crate | 职责 |
|---|---|
| `ft-cli` | `control` / `spawn` / `bench-*` |
| `ft-server` | HTTP 反代、粘滞、前缀表 |
| `ft-moe` / `ft-bench` | q* 放置画像（纯函数） |

自研 decode / 自研 CUDA 层已移除；不要在本仓库里再焊 `model.forward`。
