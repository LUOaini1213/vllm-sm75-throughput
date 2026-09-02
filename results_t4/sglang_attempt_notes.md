# SGLang 对照：五个环境坑，四个已解，第五个未完成

目标是在同硬件同模型下对照 vLLM 与 SGLang（含 `--enable-prefix-caching` 对
`--disable-radix-cache`）。**未取得可用数据**，原因记录如下，供在 `sm_75` /
小显存卡上部署 SGLang 的人参考。

## 环境

- 本地：GTX 1650（4 GiB，`sm_75`，无 Tensor Core），WSL2 Ubuntu 24.04，Python 3.12
- 云端：Colab T4，Python 3.13
- SGLang 0.5.18，torch 2.13.0+cu130

## 逐个坑

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | `Failed building wheel for outlines_core` | SGLang 依赖 `outlines_core`（Rust 实现），PyPI 无 **Python 3.13** 轮子；Colab 是 3.13，无 Rust 工具链故源码编译失败 | 改用本地 WSL 的 **Python 3.12**，有预编译轮子 |
| 2 | 启动后 `sigquit from a child process` | 默认 attention 后端 `flashinfer` 在 `sm_75` 上支持不全 | `--attention-backend triton` |
| 3 | 捕获 42 个 CUDA Graph 时子进程被杀，`avail_mem=0.32 GB` | 4 GiB 卡上图捕获耗尽显存 | `--disable-cuda-graph` + `--mem-fraction-static 0.55` |
| 4 | `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | Triton 后端需现场编译内核，WSL 只有 torch 自带 CUDA 运行时 | `apt install nvidia-cuda-toolkit`，并显式 `export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit` |
| **5** | `5 errors detected in the compilation of flashinfer/data/csrc/norm.cu` → `Ninja build failed` | 即便 attention 改 triton，**sampling 后端仍是 flashinfer**，其 CUDA 源码需较新 CUDA；apt 提供的是 **nvcc 12.0**，与 torch 的 cu130 不匹配 | **未解**。需装 CUDA Toolkit 13.x（约 4 GB） |

服务本身能起来（`[radix_on] 就绪，用时 112s`，KV cache 67,806 tokens），
但第一次真实推理请求触发 flashinfer 内核编译即崩溃，压测脚本只拿到空表。

云端并行的那条线在坑 1 修复后，因 Colab 免费额度触及运行时长上限而断开，同样未出数。

## 顺带确认的一处框架差异

`--dtype float32` 传给 SGLang 后，日志仍显示：

```
Compute capability below sm80. Use float16 due to lack of bfloat16 support.
KV Cache is allocated. dtype: torch.float16
```

即在 `sm_75` 上 KV cache 精度被覆盖为 fp16。对照之下 vLLM 接受 `--dtype float32`
并实测正常（同卡并发 16 达 295.4 tok/s）。

**这一条只是启动日志的观察，不是压测结论** —— 因为 SGLang 从未跑出一条完整请求，
无法比较两者的实际吞吐。要下"vLLM 在无 Tensor Core 卡上更可配"这种判断，
需先把坑 5 解决、拿到两边可比的数。**在此之前不作该结论。**

## 结论

在 `sm_75` + 4 GiB 上把 SGLang 跑起来是可行的，但需要：
Python ≤3.12、triton attention 后端、关 CUDA Graph、CUDA Toolkit 13.x。
本轮止步于最后一项。
