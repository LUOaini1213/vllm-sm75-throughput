# vLLM 在 GTX 1650 (sm_75) 上的部署与吞吐分析

> ### ✅ 验证状态（2026-09-02 已确认）
>
> 第四节把吞吐塌陷归因于「`sm_75` 缺少 Tensor Core 时 cuBLAS 的 fp16 GEMM 小 M 路径病态」。
> **该归因已由受控对照实验确认。**
>
> 在 **Tesla T4** 上重跑同一份 `gemm_cliff.py`。T4 与 GTX 1650 同为 `sm_75`、同为 Turing，
> **唯一差别是 T4 保留了 Tensor Core，而 GTX 16 系将其移除**：
>
> | fp16 `[M,896]@[896,4864]` | M=1 | M=2 | 相对 M=1 |
> |---|---|---|---|
> | GTX 1650（**无** Tensor Core） | 99.5 GiB/s | **4.3 GiB/s** | **0.04×** |
> | Tesla T4（**有** Tensor Core） | 106.7 GiB/s | **88.1 GiB/s** | **0.83×** |
>
> T4 上塌陷完全消失，三组形状一致（0.83× / 0.85× / 0.86×）。
> 两卡 M=1→2 的比值相差约 **20 倍**，控制变量唯有 Tensor Core 的有无 —— 归因成立。
>
> 反向佐证：T4 上 fp32 在大 M 时衰减**快于** fp16（M=64 时 0.39× vs 0.74×），
> 即「有 Tensor Core 时 fp16 更耐受大 M」，与 GTX 1650 上的表现完全相反。
>
> 复现：`cloud_vllm_verify.ipynb` / `gemm_cliff.py`，Colab T4 一键运行。
>
> **已上报上游：**
> - PyTorch（根因所在）：https://github.com/pytorch/pytorch/issues/195716
> - vLLM（部署陷阱与告警建议）：https://github.com/vllm-project/vllm/issues/54950


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

## 附：前缀复用（已在 T4 上完成受控对照）

本地 GTX 1650 那组数据因处于 fp16 GEMM 塌陷状态而**判定无效** —— 解码侧瓶颈把
prefill 的差异完全淹没了。已在 Colab T4（fp16 正常）重做，除 `--enable-prefix-caching`
外两次配置完全一致：

| 场景 | 开 prefix-caching | 关 prefix-caching |
|---|---|---|
| 无共享前缀（对照） | 1225.4 tok/s ・ 57.9 ms | 1009.5 tok/s ・ 60.8 ms |
| **共享前缀 2380 字符** | 989.5 tok/s ・ **79.4 ms** | 967.5 tok/s ・ **122.5 ms** |

- **TTFT 降低 43.1 ms（−35%）**，吞吐仅 +2.3%（噪声量级）。
- 收益出现在 TTFT 而非吞吐，与原理一致：命中 KV cache 省掉的是重复 **prefill**，
  decode 阶段的工作量并未减少。
- 「无共享前缀」两组相差 <3 ms，说明上面 43 ms 的差值来自该特性本身而非运行间噪声。
- 幅度受限于本配置（0.5B 模型 prefill 本就便宜、前缀仅 2380 字符）；
  前缀更长或模型更大时收益应更显著，**但本轮未测，不作外推**。

原始数据：[`results_t4/t4_serving_and_prefix.txt`](results_t4/t4_serving_and_prefix.txt)

---

# 2026-09-05 追加：四轮受控实验

前面的内容到 09-02 为止。这一节是 09-05 补的四轮，每轮的**判据都在跑之前写死**，
结果不管好看不好看都照原样记。全部在同一张 Colab T4 上，压测脚本
`bench_serving.py` 逐字节未改（`sha256` 前 16 位 `ef5bf10717ecdacf`），所以横向可比。

## A. SGLang vs vLLM：前缀复用「开 vs 关」同口径对照

之前 vLLM 只有「同一服务内有无共享前缀」的差（TTFT −35%），
而 SGLang 有「特性开 vs 关」的差 —— **两个口径不同，并列比大小是错的**。
于是给 vLLM 补跑了关闭 `--enable-prefix-caching` 的那一组。

同口径（共享前缀场景，开 ÷ 关）：

| | 吞吐 | TTFT p50 |
|---|---|---|
| vLLM | 861.9 → 1016.8 = **1.18×** | 169.2 → 65.5 ms = **−61.3%** |
| SGLang | 540.3 → 832.7 = **1.54×** | 464.9 → 123.6 ms = **−73.4%** |

**但只报收益比会误导**：关闭时 vLLM 的绝对表现明显更好（861.9 vs 540.3 tok/s）。
SGLang 收益比更大，部分原因是它的关闭态基线更差。**两组绝对值必须一起给。**

另一条更清楚的差别 —— **没有**共享前缀可复用时：
vLLM 几乎不收费（TTFT 49.2 → 48.3 ms），**SGLang 要付 45% 的 TTFT 代价**（69.4 → 100.9 ms）。

→ [`results_t4/prefix_control_vllm_vs_sglang_2026-09-05.txt`](results_t4/prefix_control_vllm_vs_sglang_2026-09-05.txt)

## B. 量化：一个被数据推翻的预测

跑之前写下：「T4 是 sm_75，vLLM 的 GPTQ 快速路径 Marlin 要 sm_80 以上，
所以 **GPTQ-Int4 的吞吐可能不如 fp16**。」

**预测错了。** Qwen2.5-1.5B 上，两种 PTQ 在全并发区间都快于 fp16：

| 并发 | fp16 | AWQ | GPTQ-Int4 |
|---|---|---|---|
| 1 | 66.3 | 124.9 (1.88×) | 126.2 (**1.90×**) |
| 8 | 419.6 | 787.2 (1.88×) | 761.2 (1.81×) |
| 32 | 1163.3 | 1703.9 (1.46×) | 1706.8 (1.47×) |

前提（sm_75 确无 Marlin，第 0 节实测确认）对，**推论错**——我默认了 kernel 效率是瓶颈却没验证。
原始预测原文保留在结果文件里，没有事后修改。

值得单独讲的取舍：**TPOT 腰斩**（14.89 → 7.7 ms）、**KV cache +13%**，
**但 TTFT 反而变差**（55.2 → 65~69 ms）—— 量化省的是权重读取，prefill 计算量没变、
反量化还要额外开销。

**边界**：量化权重用的是 Qwen 官方已量化版本，**量化过程不是本仓做的**；**QAT 未做**；
精度只有代理指标（与 fp16 的贪心输出一致率），**PPL 与 LongBench 均未跑**。

→ [`results_t4/quant_fp16_awq_gptq_2026-09-05.txt`](results_t4/quant_fp16_awq_gptq_2026-09-05.txt)

## C. 那 GPTQ 为什么反而更快？—— roofline 检验

B 里给的解释「解码受权重读取带宽限制」当时**只是假说、无证据**。这一轮补证据。

不用标称的 320 GB/s，实测可达带宽：copy 241.4 / **read 275.9** / triad 238.9 GB/s。
解码读权重是纯读，用 read 当屋顶。

| 变体 | 权重 | 理论下界 | 实测 TPOT | 实测/下界 |
|---|---|---|---|---|
| fp16 | 2.98 GiB | 11.60 ms | 14.58 ms | **1.26** |
| AWQ | 1.10 GiB | 4.28 ms | 7.20 ms | **1.68** |
| GPTQ-Int4 | 1.10 GiB | 4.28 ms | 7.25 ms | **1.69** |

离散度 1.35 ≤ 跑前写死的阈值 1.6 → **假说成立**。屋顶只算权重、没算 KV cache 与激活，
还能贴到 1.26–1.69 倍，说明 batch=1 解码确由权重读取主导。

**比「成立」更有信息的一条**：权重之比 2.71×，实测 TPOT 之比只有 2.03×，
**只吃到理论访存收益的 75%**；且 fp16 离屋顶更近（1.26×）、量化版更远（1.68×）。
差额方向指向反量化开销 —— **这是从残差反推，不是直接测量**，要坐实需 ncu 级剖析。

顺带测到「带宽受限 → 算力受限」的拐点：并发 1→8 时 TPOT 几乎持平、量化优势稳定约 2×；
并发 16 起 TPOT 抬头、优势收窄到 1.5×。**拐点在并发 8 与 16 之间。**

→ [`results_t4/roofline_decode_2026-09-05.txt`](results_t4/roofline_decode_2026-09-05.txt)

## D. SGLang 一个「凹陷」的根因 —— 以及一次自我更正

先前记录 SGLang 在并发 16 处吞吐塌到并发 8 的 42%，三轮重复 3/3 复现，
当时写成「并发 16 是真实凹陷」。**这个归因是错的。**

**真正的根因：解码 CUDA graph 只捕获到 batch size 8。** 启动日志原文：

```
Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1,
bs=[1, 2, 4, 8], avail mem=1.91 GB
```

batch > 8 没有图可用、回落 eager，**每 token 解码延迟一步跳约 4.8 倍后保持平坦**：

| batch | 8 | 10 | 12 | 14 | 16 | 18 | 32 |
|---|---|---|---|---|---|---|---|
| TPOT p50 (ms) | **6.23** | **29.76** | 29.76 | 28.51 | 31.01 | 31.52 | 31.74 |

台阶在 **8 → 10 之间**，不在 16。之所以表现为「并发 16 凹陷」，是因为吞吐 ≈ batch / TPOT：
8→16 batch 翻倍而 TPOT 涨 4.7 倍 → 吞吐掉一半；16→32 batch 再翻倍而 TPOT 不变 → 吞吐回升。
**「凹陷」是台阶与扫描列表 batch 倍数相除的产物，不是 16 这个数字本身有问题。**

同一轮里排除掉的两条：

- **调度/排队**：并发 8 与 16 的 `#queue-req` 全为 0、`#running-req` 恰等于请求并发、
  `preempt` 0 次 → 排除。
- **chunked prefill**：`--chunked-prefill-size` 取 512 / 2048 / 8192，16/8 比值
  0.46 / 0.48 / 0.58，**凹陷纹丝不动** → 排除。

**未完成**：默认 flashinfer 后端那组服务 420s 内没起来，**不能声称结论适用于默认路径**。

**证据强度**：目前建立在「日志写明 `bs=[1,2,4,8]`」+「TPOT 台阶恰在 8 之后」两条吻合上，
属**强关联**，不是因果实证。可证伪的下一步：提高 `--cuda-graph-max-bs`，台阶应当右移。
**本仓尚未跑这一步。**

→ [`results_t4/dip_rootcause_2026-09-05.txt`](results_t4/dip_rootcause_2026-09-05.txt)

## 复现

六份 notebook 可直接在 Colab（T4）打开跑，每份第 1 节都列了该框架的环境阻塞与修法：

| notebook | 做什么 |
|---|---|
| `cloud_sglang_v2.ipynb` | SGLang vs vLLM 受控对照（含 SGLang 在 Colab 的四个依赖阻塞修法） |
| `cloud_vllm_prefix_control.ipynb` | vLLM 前缀复用开/关，补齐口径 |
| `cloud_quant_matrix.ipynb` | fp16 / AWQ / GPTQ-Int4 |
| `cloud_roofline_decode.ipynb` | 带宽屋顶检验 |
| `cloud_sglang_repeat.ipynb` | 离群值重复 3 轮 |
| `cloud_dip_rootcause.ipynb` | 凹陷根因四实验 |

`tools_check_notebooks.py` 是配套的 notebook 静态检查，六条规则各对应一次真实踩坑
（嵌套引号吞掉字符串、非法 `%` 格式符、正则缺捕获组配 `group(1)`、循环里的宽 `except` 等）。
