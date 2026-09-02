# vLLM 在 GTX 1650 (sm_75) 上的部署与吞吐分析

> ### ⚠️ 验证状态（2026-09-02）
>
> 本文第四节把吞吐塌陷归因于「`sm_75` 缺少 Tensor Core 时 cuBLAS 的 fp16 GEMM 小 M 路径病态」。
> **这是一个推论，尚未被证实。**
>
> 已有证据：在 GTX 1650（`sm_75`，**无** Tensor Core）上，fp16 GEMM 于 M=1→2 塌陷 23 倍，
> 同形状 fp32 无塌陷；vLLM 层面 fp32 批量吞吐最高快 9.1 倍。
>
> **缺的关键对照**：同为 `sm_75` 但**带** Tensor Core 的卡（T4 / RTX 20 系）上是否也塌陷。
> - 若不塌陷 → 归因成立。
> - 若同样塌陷 → **归因被推翻**，本文第四、五节须改写。
>
> 该对照实验已写成 `cloud_vllm_verify.ipynb` 与 `gemm_cliff.py`，可在 Colab T4 上一键复现。
> 结论确认前，请把第四节当作**待验证假设**而非结论。


在一张 **GTX 1650（4 GiB，Turing sm_75，无 Tensor Core）** 上把 vLLM 0.28.0 跑起来，
压测并发吞吐，过程中发现并定位了一个 **fp16 批量推理吞吐塌陷 23 倍** 的问题。

环境：Windows 11 + WSL2 Ubuntu 24.04.4（内核 6.18.33）· vLLM 0.28.0 ·
模型 Qwen2.5-0.5B-Instruct（hidden 896 / intermediate 4864）。

---

## 一、部署：四个真实的坑

vLLM 没有 Windows 轮子，只能走 WSL2。服务起了四次才成功，每次的原因都不一样：

| # | 报错 | 原因 | 处理 |
|---|---|---|---|
| 1 | `Free memory (3.22/4.0 GiB) < desired utilization (0.85, 3.4 GiB)` | Windows 桌面合成器占了 0.78 GiB，4 GiB 卡实际只有 3.22 GiB 可用 | `--gpu-memory-utilization 0.75` |
| 2 | `RuntimeError: UVA is not available` | vLLM 在 WSL 下**默认关闭锁页内存**，而 v0.28 的 `UvaBuffer` 硬依赖它 | 本机内核 6.18.33 ≫ vLLM 要求的 4.19.121，加 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` |
| 3 | `Failed to find C compiler` | Triton 要现场编译内核，WSL 的 Ubuntu 精简镜像没带编译器 | `apt install build-essential` |
| 4 | `Python.h: No such file` | 同上，缺 CPython 开发头文件 | `apt install python3.12-dev` |

启动成功后的关键日志：

```
Using TRITON_ATTN attention backend out of potential backends:
    ['TRITON_ATTN', 'FLEX_ATTENTION']
Available KV cache memory: 1.7 GiB
GPU KV cache size: 148,576 tokens
Maximum concurrency for 2,048 tokens per request: 72.55x
init engine (profile, create kv cache, warmup model) took 102.93 s
```

**`FlashAttention` 连候选列表都没进** —— 它要 sm_80+，这张卡是 sm_75，
vLLM 直接回落 `TRITON_ATTN`。这是「同一份 vLLM，换张卡就是另一套性能」最直接的证据。

---

## 二、异常：并发从 1 变成 2，吞吐掉 10 倍

首轮并发扫描（fp16）：

| 并发 | 墙钟 s | 吞吐 tok/s | TTFT p50 | TPOT p50 |
|---|---|---|---|---|
| 1 | 17.08 | 119.5 | 147.4 ms | **7.20 ms** |
| 2 | **185.53** | **11.0** | 234.2 ms | **180.44 ms** |
| 4 | 93.13 | 22.0 | 317.4 ms | 180.97 ms |
| 8 | 93.27 | 43.9 | 322.6 ms | 181.06 ms |
| 16 | 95.65 | 85.6 | 652.1 ms | 183.89 ms |
| 32 | 98.49 | 165.7 | 751.2 ms | 184.94 ms |

并发 2 跑同样 16 条请求，墙钟是并发 1 的 **10.9 倍**。
服务端自己的日志给出同样的数（`Running: 2 reqs, generation throughput: 11.0 tokens/s`，
`GPU KV cache usage: 0.3%`，`Waiting: 0 reqs`），所以不是客户端的问题。

TPOT 在并发 2→32 之间恒定在 180~185 ms —— **每步耗时与实际 batch 无关**，
这是「有个固定开销在主导」的典型特征。

---

## 三、定位：逐个排除

| 假设 | 实验 | 结果 |
|---|---|---|
| 压测客户端有 bug | 读服务端自身 metrics | ❌ 服务端报同样的数 |
| KV cache 不够 / 请求排队 | 看 `GPU KV cache usage` / `Waiting` | ❌ 用量 0.3~1.0%，无排队 |
| GPU 降频、过热、功耗墙 | 负载中采样 `nvidia-smi` | ❌ SM 1785 MHz 满血，所有 throttle 标志 Not Active |
| 显存带宽打满 | 同上 | ❌ 显存控制器仅 4% 占用（但 SM 100%）→ 纯算力侧问题 |
| CUDA Graph / torch.compile 只对 batch=1 生效 | `--enforce-eager` 对照 | ❌ batch≥2 两者都是 ~183 ms（CUDA Graph 只在 batch=1 有用，7.20 vs 19.13 ms） |
| 按 `--max-num-seqs` padding | 32 → 4 | ❌ 仍是 180.99 ms |
| 按 KV cache 全表扫描 | 150,288 → 59,856 tokens | ❌ 仍是 180.16 ms |
| 新请求 prefill 插队拖累解码 | 一次性发 N 条、期间无新到达 | ❌ 仍塌陷 |

**纯稳态突发测试**（无新请求到达，稳态就是 batch=N）：

| 一次性并发 | 墙钟 | 吞吐 tok/s | TPOT p50 |
|---|---|---|---|
| 1 | **0.58 s** | 109.5 | 6.98 ms |
| 2 | **7.77 s** | 13.3 | 151.76 ms |
| 3 | 7.79 s | 17.4 | 182.88 ms |
| 4 | 7.81 s | 22.2 | 183.17 ms |
| 8 | 7.81 s | 41.6 | 183.43 ms |
| 16 | **7.81 s** | 80.5 | 183.47 ms |

**墙钟从 N=2 到 N=16 精确锁死在 7.8 秒。** 吞吐线性上升纯粹因为同样时间产出更多 token。

换算成权重读取带宽（0.5B 模型每 step 约读 1 GiB 权重）：

- batch=1：9 ms/step → **约 111 GB/s**，接近该卡带宽上限
- batch≥2：122 ms/step → **约 8 GB/s**，仅峰值的 6%（与观察到的「显存 4% 占用」吻合）

batch=1 是 **GEMV**（矩阵×向量），batch≥2 变成 **GEMM**（矩阵×矩阵）。假设指向 GEMM 内核选择。

---

## 四、根因：sm_75 无 Tensor Core 时 fp16 GEMM 的小 M 塌陷

脱离 vLLM，直接测裸 PyTorch 矩阵乘（`gemm_cliff.py`，形状取自模型实际层）：

**fp16 · `[M,896] @ [896,4864]`（MLP gate/up）**

| M | 耗时 ms | 权重带宽 GiB/s | 相对 M=1 |
|---|---|---|---|
| **1** | 0.0816 | **99.5** | 1.00× |
| **2** | 1.8708 | **4.3** | **0.04×** |
| 4 | 1.7943 | 4.5 | 0.05× |
| 16 | 1.8349 | 4.4 | 0.04× |
| 64 | 1.7943 | 4.5 | 0.05× |

**fp32 · 同样形状 —— 没有塌陷**

| M | 耗时 ms | 权重带宽 GiB/s | 相对 M=1 |
|---|---|---|---|
| 1 | 0.1411 | 115.0 | 1.00× |
| **2** | 0.1381 | **117.6** | **1.02×** |
| 8 | 0.2159 | 75.2 | 0.65× |
| 16 | 0.3816 | 42.5 | 0.37× |

另两组形状（MLP down、QKV proj）呈现完全一致的模式。

**结论：GTX 16 系是 Turing 架构但移除了 Tensor Core。**
cuBLAS 在 sm_75 上的 **fp16 GEMM 小 M 路径是病态的**（M≥2 即塌到 4~5 GiB/s），
而 fp16 GEMV（M=1）和所有 fp32 路径都正常。vLLM 的 batch 悬崖只是这一底层现象的放大。

---

## 五、验证：反直觉的预测

如果根因成立，那么**在这张卡上批量推理用 fp32 应该比 fp16 快**。用 `--dtype float32` 重跑：

| 一次性并发 | fp16 吞吐 | **fp32 吞吐** | 提升 | fp32 TPOT |
|---|---|---|---|---|
| 1 | 109.5 | 82.5 | 0.75× | 11.66 ms |
| 2 | 13.3 | **85.2** | **6.4×** | 12.11 ms |
| 3 | 17.4 | **155.9** | **9.0×** | 14.43 ms |
| 4 | 22.2 | **203.1** | **9.1×** | 13.89 ms |
| 8 | 41.6 | **274.5** | **6.6×** | 21.57 ms |
| 16 | 80.5 | **295.4** | **3.7×** | 43.81 ms |

**预测成立，最高 9.1 倍。** 而且 fp32 呈现出教科书式的 continuous batching 曲线：
吞吐 82.5 → 295.4（3.6×），TPOT 11.66 → 43.81 ms 平滑上升 —— 这才是正常的吞吐-延迟取舍。
fp16 只在 batch=1 时有优势。

**代价**：fp32 权重占 2 GiB，KV cache 从 148,576 tokens 缩到 **5,824 tokens**（`--max-model-len 1024`）。
在这张卡上，「批量吞吐」和「上下文容量」之间必须二选一。

---

## 六、复现

```bash
# 一次性环境准备（WSL2 Ubuntu）
bash setup.sh

# 起服务（fp16 默认配置）
bash serve.sh

# 并发扫描 + 前缀复用对照
bash run_bench.sh

# 稳态突发测试（无新请求到达）
python isolate.py

# 裸 GEMM 微基准 —— 根因所在
python gemm_cliff.py

# dtype 对照
bash _exp_dtype.sh float32 1024 fp32
```

| 文件 | 说明 |
|---|---|
| `setup.sh` | WSL 环境准备：uv + vLLM + 模型 |
| `serve.sh` / `serve_eager.sh` | 服务启动（含四个坑的处理）/ 关 CUDA Graph 对照组 |
| `bench_serving.py` | 并发扫描，测 TTFT / TPOT / 吞吐 |
| `isolate.py` | 稳态突发测试，隔离 prefill 插队的影响 |
| `gemm_cliff.py` | 裸 GEMM 微基准，定位根因 |
| `_exp.sh` / `_exp_dtype.sh` | 参数化对照实验 |

---

## 附：前缀复用

`--enable-prefix-caching` 下 2380 字符共享 system prompt 的对照，
在 fp16 配置上未观察到 TTFT 改善（321 ms → 499 ms）。
考虑到该配置本身处于上述 GEMM 塌陷状态，prefill 的收益被解码侧的瓶颈完全淹没，
这组数据**不足以评价前缀复用本身**。要得到有意义的结论，
需在 fp32 配置下、并对照关闭 `--enable-prefix-caching` 重测。**留作待办。**
