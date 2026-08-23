# 编排层怎么接 FreeToken

> 2026-08-23  
> 结论：blend 不是唯一选择；**Nebula 接 `ft serve` 同样可行**，而且和现有 vLLM/SGLang 编排是一类事。

---

## 1. FreeToken 对编排器暴露什么

一个 `ft serve` = **一份模型**（或一组 TP 卡）+ **一个 HTTP 端口**（OpenAI / Anthropic 兼容）。

编排器只需要：

1. 拉起进程或容器（镜像内 `freetoken`，或主机 venv + `CUDA_VISIBLE_DEVICES`）  
2. 把请求打到 `--host/--port`  
3. 就绪探测：`GET /v1/models` 或日志 `API server is ready to serve`  
4. **同一会话必须钉在同一个 `ft` 实例上**（KV / radix 只活在该进程里）

不需要改 FreeToken 源码，也不走 Python import。

---

## 2. 和 blend 是同一层

```
客户端
   │
   ▼
编排（Nebula 或 blend-control）
   ├─ vllm serve
   ├─ sglang.launch_server
   └─ ft serve --model … [--tp-size 2]
```

| | blend-control | Nebula 接 FT |
|---|---|---|
| 角色 | 薄 HTTP 网关 + 粘滞 | 已有的多引擎编排（etcd、docker、抢卡） |
| 和 vLLM/SGLang | 不管 | 本机已经在管 |
| 接 FT | `--worker http://ft:port` | 加一种 backend 规格 |

**单模型、单 `ft serve`：** 两者都几乎没增量（FT 自己就能 serve）。  
**一台 8 卡上多模型 / 多副本 / 长会话：** 编排才有存在感。

---

## 3. 为什么说 Nebula 更顺

本机已经有：

- `nebula-node` + etcd  
- `nebula-qwen15_moe_vllm-0` / `nebula-qwen15_moe_sglang-0`  

要的是「再承认一种引擎」，不是再养一个只懂 FT 的网关。放置、和 cortex 抢卡、容器生命周期，都在 Nebula 里更完整。

blend 仍适合：不动 Nebula、只要静态反代、或单独试 FT 粘滞/前缀。

---

## 4. 接入时不要套错模型

FreeToken **不是**「一个引擎里挂很多模型」：

- 一进程一份权重（`--tp-size N` 也只是这一份切 N 卡）  
- 多模型 = 多个 `ft serve`  
- 会话 / 前缀必须粘在**同一个**进程，否则每轮整段 re-prefill  

规格示例：

```text
backend: freetoken
command: ft serve --model <hf_or_dir> --host 0.0.0.0 --port $PORT [--tp-size 2]
env:
  CUDA_VISIBLE_DEVICES: "0,1"
health: GET /v1/models
sticky: session_id → 该实例
```

NCCL：venv 里常只有 `libnccl.so.2`，JIT `pynccl` 需要 `libnccl.so` 符号链接。

---

## 5. 和 blend 命令的对应

若继续用 blend 当薄网关：

```bash
ft serve --model … --port 1930          # 仍是 FreeToken 自己拉起
blend control --worker http://127.0.0.1:1930
```

Nebula 做同一件事时，把上面的 `ft serve` 收进它的 backend 表即可，**不必经过 blend**。
