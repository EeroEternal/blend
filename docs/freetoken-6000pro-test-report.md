# FreeToken 在 RTX PRO 6000 8 卡机上的部署与测试报告

> 测试日期：2026-08-22
> 测试人：ox-alpha
> 论文：[FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://arxiv.org/abs/2608.16157) (arXiv:2608.16157)
> 代码：https://github.com/FlashML-org/FreeToken
> 主页：https://www.flashml.ai/

---

## 1. 结论速览

在 8×RTX PRO 6000 Blackwell（96GB）工作站上，FreeToken v0.1.2 **成功跑通**两个目标模型：

| 模型 | 参数量 | 精度 | GPU / 端口 | 专家池加载 | 稳态解码速度 | 状态 |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 | 284B | ds_fp4 | GPU 4 / :1919 | 143GB / 66s | **≈31–32 tok/s** | ✅ |
| GLM-5.2-NVFP4 | 753B | NVFP4 | GPU 5 / :1921 | 419GB / 3.4min | **≈16 tok/s** | ✅ |

两个模型实例可**同时并存**运行，主机内存合计占用约 563GB（总 755GB），仍有约 184GB 余量。

---

## 2. FreeToken 原理概述

FreeToken 是 Edge-Native MoE 推理引擎，把个人工作站视为 GPU + CPU + DRAM + PCIe 的统一弹性推理平台，而非"一块小 GPU"。核心设计为两大执行原则 + 一个资源管理策略：

### 2.1 带宽自适应执行

- **Prefill 全层双缓冲**：GPU 计算第 `l` 层时，第 `l+1` 层专家经 PCIe 流式预加载。Prefill 会激活几乎整个专家池（数千 token 的路由覆盖全部专家），双缓冲将该 I/O 开销隐藏在计算之后。
- **Decode 的 q\* 策略**：每个 token 路由 top-k 专家。GPU 上维护全层共享的 LRU 专家缓存：
  - 缓存命中 → 直接 VRAM 计算；
  - miss 拆分为两部分：一部分经 PCIe 取回 GPU 填缓存，其余**留在 CPU 内存原地计算**；
  - 拆分比例 `q* = m · B_PCIe / B_Host`，由每台机器实测带宽决定（PCIe 与 CPU 专家计算共享同一内存带宽子系统，静态 offload 必然失衡）。

### 2.2 语义感知缓存

- Prefill 在特殊 token 锚点（`<think>`、`</tool_call>`、`</tool_output>` 等）保存 recurrent-state 检查点；agent 编辑上下文后只需对修改点之后的后缀重新 prefill。
- Decode 阶段相邻 token 路由高度重叠（时间局部性），LRU 缓存命中率高，残余 miss 才进入 q\* 路径。

### 2.3 弹性内存管理

- 运行时在调度安全点动态重分配 VRAM（专家缓存 ↔ KV cache），无需重启引擎或重载权重。
- 权重直接加载进最终主机布局后再 pin 内存，加速启动。

### 2.4 MoE 后端

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`：

| 后端 | 行为 |
|---|---|
| `fused` | 专家常驻 GPU（需要足够 VRAM） |
| `offload` | 专家驻留主存，GPU LRU 缓存，miss 走 PCIe |
| `cpu` | miss 全部由 CPU 就地计算 |
| `hybrid` | 每步按 q\* 比例拆分：部分 PCIe 取回 + 部分 CPU 就地算，重叠执行 |
| `auto` | 由 `ft bench bw` 的硬件画像自动决定 |

---

## 3. 测试环境

### 3.1 硬件

| 项目 | 配置 |
|---|---|
| GPU | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition（96GB each） |
| CPU | AMD EPYC 9355 32 核（128 线程） |
| 内存 | 755GB DDR5 |
| 磁盘 | 3.5TB NVMe（可用 2.4TB） |

**注意**：测试时 GPU 0–3、6–7 被 vLLM / SGLang 生产任务占用，实际使用空闲的 **GPU 4 和 GPU 5**。

### 3.2 软件

| 项目 | 版本 |
|---|---|
| 系统 | Linux x86_64 |
| 驱动 | 580.65.06（满足 FreeToken 要求的 r580+） |
| CUDA Toolkit | 13.0（`/usr/local/cuda-13.0`，JIT 编译内核所需） |
| Python | 3.10.12 |
| FreeToken | 0.1.2（PyPI 安装，`pip install "freetoken[accel]"`） |

---

## 4. 部署过程

### 4.1 安装

```bash
# PyPI 直连过慢，改用清华镜像
python3 -m venv ~/freetoken-venv
source ~/freetoken-venv/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "freetoken[accel]"
```

安装包含 torch 2.11.0 (cu13)、flashinfer-python 0.6.17、sglang-kernel 0.4.5 等，CUDA 内核首次使用时 JIT 编译（需 `nvcc` 在 PATH 中）。

### 4.2 带宽画像标定

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
CUDA_VISIBLE_DEVICES=4 ft bench bw --model dsv4,glm4.7-nvfp4
```

实测结果：

```
host pro6000   gpu cuda:0 (NVIDIA RTX PRO 6000 Blackwell Server Edition)   cpu 64c/64t
ceilings: CPU STREAM read 124.4  |  PCIe linear H2D 57.7  D2H 57.4  GB/s   (threshold 2.0x)

dsv4  H=4096 I=2048 E=256 top_k=6
  ds_fp4    12.75 MB    CPU-MoE 155.4 GB/s   PCIe-gather 52.4 GB/s   2.96x → hybrid
  overlapped: CPU-MoE 152.2 + PCIe 34.4 GB/s → hybrid fetches 18.4% of misses

glm4.7-nvfp4  H=5120 I=1536 E=160 top_k=8
  nvfp4     12.67 MB    CPU-MoE 155.0 GB/s   PCIe-gather 52.9 GB/s   2.93x → hybrid
  overlapped: CPU-MoE 148.3 + PCIe 34.1 GB/s → hybrid fetches 18.7% of misses
```

CPU/PCIe 带宽比 2.96x > 阈值 2.0，引擎自动推荐 **hybrid 后端**——即约 18% 的缓存 miss 走 PCIe 取回 GPU，其余 82% 在 CPU 就地计算。

### 4.3 DeepSeek-V4-Flash 启动

```bash
CUDA_VISIBLE_DEVICES=4 ft serve --model ~/models/DeepSeek-V4-Flash-0731 --host 0.0.0.0 --port 1919
```

关键日志：

```
Auto-selected attention backend: dsv4_sparse
Auto-selected MoE backend: hybrid
Loading experts (parallel): 143G [01:06, 2.64GB/s]        ← 66 秒完成
moe_cache_size=5835 num_pages=501 (prefill_overlap=True)
fetching 18.4% of each decode step's expert misses over PCIe
CPU MoE executor ready: threads=63 (pinned to cores 0..62) isa=avx512bf16
Allocating 64128 tokens for DSV4 KV cache
API server is ready to serve on 0.0.0.0:1919
```

### 4.4 GLM-5.2 权重下载

HF 直连被墙（SSL EOF），hf-mirror 的 xet 桥也不稳定（CAS 端点 401 / 超时）。最终从 ModelScope 镜像下载：

```bash
modelscope download --model nv-community/GLM-5.2-NVFP4 --local_dir ~/models/GLM-5.2-NVFP4
```

- 总量 464.9GB（47 个 safetensors 分片）
- 实测速度 ~70–80MB/s，全程约 100 分钟
- 若必须走 hf-mirror：需设置 `HF_HUB_DISABLE_XET=1` 回退到普通 HTTP

### 4.5 GLM-5.2 启动

```bash
CUDA_VISIBLE_DEVICES=5 ft serve --model ~/models/GLM-5.2-NVFP4 --host 0.0.0.0 --port 1921
```

关键日志：

```
Resolved config: moe_backend='hybrid', attention_backend='dsa', cache_type='radix'
Loading experts (parallel): 419G [03:23, ~2GB/s]           ← 419GB 专家池 3.4 分钟
moe_cache_size=3226 num_pages=8304
fetching 18.7% of each decode step's expert misses over PCIe
CPU MoE executor ready: threads=63 isa=avx512bf16+avx512vnni(nvfp4-w4a8)
  H=6144 I=2048 experts=256 layers=75 top_k=8
API server is ready to serve on 0.0.0.0:1921
```

> 踩坑记录：初次用端口 1920 启动失败（该端口已被其他服务占用，`Errno 98`），换 1921 成功。

---

## 5. 性能实测

### 5.1 DeepSeek-V4-Flash-0731（284B）

请求示例（中文问答，max_tokens=400）：

```
Decode batch, #running-req: 1, gen throughput (token/s): 31.62
Decode batch, #running-req: 1, gen throughput (token/s): 31.80
Decode batch, #running-req: 1, gen throughput (token/s): 32.00
```

- 端到端（含 prefill）：214 completion tokens / 49.8s ≈ 4.3 tok/s（首个请求含预热）
- **稳态解码：≈31–32 tok/s**

对比论文数据（RTX 5090 上 22–25 tok/s）：本机更快，主要归因于 EPYC 平台 124.4 GB/s 的内存带宽（高于典型桌面 DDR5 双通道的 80–90 GB/s），CPU 就地计算路径直接受益。

### 5.2 GLM-5.2-NVFP4（753B）

请求示例（英文长文生成，max_tokens=500）：

```
Decode batch, #running-req: 1, gen throughput (token/s): 15.74~16.00
```

- 一次 399-token 中文回答端到端耗时 35.6s（≈11.2 tok/s，含 prefill 与 TTFT）
- **稳态解码：≈16 tok/s**

753B 模型在单张消费级卡上达到交互级速度（超过 Codex 生产环境中位解码速度 33 tok/s 的一半），验证了论文"单张 RTX PRO 6000 工作站卡跑 GLM-5.2"的结论。

### 5.3 资源占用（两模型并存）

```
Mem: total 755Gi | used 47Gi | buff/cache 683Gi | available 184Gi
GPU 4: 88649 MiB / 97887 MiB   (DSV4-Flash)
GPU 5: 88471 MiB / 97887 MiB   (GLM-5.2)
```

- 主机内存：DSV4 专家池 143GB + GLM-5.2 专家池 419GB + 基础占用 ≈ 563GB，余量充足
- 单卡 VRAM：各占约 88.7GB（非专家权重 + KV cache + 3226/5835 个专家缓存槽 + CUDA Graph）
- KV cache：GLM-5.2 分配 8304 tokens（DSA 稀疏注意力）；DSV4 分配 64128 tokens（SWA）

---

## 6. API 使用

两个服务均为 OpenAI 兼容 API（同时支持 Anthropic `/v1/messages`）：

```bash
# DeepSeek-V4-Flash
curl http://<server>:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"你好"}],"max_tokens":256}'

# GLM-5.2
curl http://<server>:1921/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.2-NVFP4","messages":[{"role":"user","content":"你好"}],"max_tokens":256}'
```

配合 coding agent 使用：`ft launch claude`（支持 claude / codex / dsh / hermes / openclaw / opencode），自动写 provider 配置并指向本地 server。

---

## 7. 经验与注意事项

1. **单卡引擎**：FreeToken 无张量并行（TP）参数，每个实例绑定一张 GPU。多卡机的价值在于可同时跑多个大模型实例。
2. **网络是最大瓶颈**：国内环境下载 465GB 权重建议优先 ModelScope 镜像；PyPI 用清华镜像。
3. **nvcc 必须**：JIT 内核编译依赖 CUDA 13 toolkit，登录 shell 默认不含 `/usr/local/cuda-13.0/bin`，需手动 export PATH。
4. **先跑 `ft bench bw`**：画像按"专家格式 + GPU 型号"缓存于 `~/.cache/freetoken/benchbw.json`，`--moe-backend auto` 依赖它决定 offload/hybrid 及 q\* 拆分比例。
5. **CPU 核心绑定**：引擎将 63 个物理核 pin 给 CPU MoE 执行器（`isa=avx512bf16+avx512vnni(nvfp4-w4a8)`），主进程 intra-op 线程降为 1，避免与专家计算争抢。同机混部其他高负载任务时注意 CPU 竞争。
6. **端口冲突**：启动前确认端口未被占用（本机 1920 已被占用导致首次失败）。
7. **内存规划**：专家池全量驻留主存（DSV4 143GB / GLM-5.2 419GB），部署前确认 `available` 内存充足。

---

## 8. 复现命令清单

```bash
ssh -p 2208 bodesi@39.183.171.3

# 环境
python3 -m venv ~/freetoken-venv && source ~/freetoken-venv/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "freetoken[accel]"
export PATH=/usr/local/cuda-13.0/bin:$PATH

# 标定
CUDA_VISIBLE_DEVICES=4 ft bench bw --model dsv4,glm4.7-nvfp4

# 服务
CUDA_VISIBLE_DEVICES=4 nohup ft serve --model ~/models/DeepSeek-V4-Flash-0731 --host 0.0.0.0 --port 1919 > ~/ft_serve_dsv4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup ft serve --model ~/models/GLM-5.2-NVFP4 --host 0.0.0.0 --port 1921 > ~/ft_serve_glm.log 2>&1 &

# 观察
tail -f ~/ft_serve_dsv4.log ~/ft_serve_glm.log
nvidia-smi; free -h
```

---

## 9. 同模型对照：blend vs FreeToken（2026-08-23 补测）

模型：`Qwen3-30B-A3B-Instruct`（BF16），GPU：同一张 RTX PRO 6000 #4。

| 引擎 | decode |
|---|---|
| FreeToken 0.1.2（offload + FlashInfer + CUDA Graph） | **111–132 tok/s** |
| blend `decode-qwen`（GPU GEMV attn + hybrid MoE） | **~9 tok/s** |

结论：同精度同权重下 blend 慢约 14×。优先补融合注意力与 CUDA Graph，而不是先做 FP4。
