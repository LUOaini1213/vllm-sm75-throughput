# 宁德训推岗位补强：可追溯的流式测量与服务验收

本目录是 2026-09-18 新做的工程练习，建立在既有 vLLM / SGLang 单卡实验上。新增实现与证据留在本目录，既有实验日志和 notebook 没有被覆盖；仓库根 README 的历史计数口径由主任务另行同步。最终结果见 `RESULTS.md` 与 `evidence/summary.json`。

实跑完成：两种精度共 **96/96 请求**、**3072 个输出 token** 双来源计数一致，**10/10 异常及恢复探测**通过；独立 generate API 同参数 **4/4** token 序列一致；7 项客户端边界测试通过。固定小题的严格匹配仅 **3/16（两种精度相同）**，保留失败与低分，不把吞吐成功当成业务准确率。

## 这次补了什么

1. **指标口径**：新增 SSE 增量解析器，记录每次请求的 token ID、最终 usage、时间戳、客户端排队、状态与响应文本。吞吐以成功请求的真实输出 token 数除以整批墙钟；失败请求不计入成功吞吐，但其耗时保留在分母。
2. **流式延迟**：TTFT 从请求发出算到第一个 token ID，另给含客户端排队的 TTFT 和首次可见文本 TTFC；只有每个 token 事件恰含一个 token 时才计算 TPOT 与 ITL，多个 token 合并成一次事件时不给伪精确的逐 token 间隔。
3. **可重现压测**：并发 1 / 2 / 4，每档 8 请求、每请求 32 输出 token、两轮；记录预热、完整请求与错误、环境和脚本哈希。此为短闭环负载，不能代表生产流量容量。
4. **兼容与恢复**：未知模型、超上下文、非法生成长度、客户端超时、客户端取消五类探测，每项后追加真实恢复请求；保留启动失败原始日志。这里只验证接下来仍可服务，不宣称服务端取消资源回收已被测量。
5. **精度评测接口**：固定 16 个小型工程情境任务，按预写答案精确匹配，可对不同 fp16 / AWQ / GPTQ 端点复用。此次只实际运行缓存 0.5B 模型的 fp16 / fp32 对照；没有新 INT4 权重，不编造量化准确率。

## 与旧结果的关系

`audit_history.py` 重新读取已执行 notebook 中 23 个历史单元，与旧检查器及转写文件核对。历史 A32 的 26.92 秒墙钟与 TTFT/TPOT 近似反推的 19.12 秒存在约 7.80 秒残差。因此旧“4.42 倍吞吐提升”不再作为干净的优化收益使用。3.48 倍对应旧流式片段计数口径的 35.74→10.26 ms，不能由本次 CPU 复算升格为新 GPU 结论。

另一个缺口是旧客户端 `n_tok += 1` 数的是非空 `delta.content` 片段，而不是来自服务端的 token ID。旧结果中的“输出约 128 / 请求”由相同计数及墙钟推导，不能独立证明片段数等于真实 token 数。本次保留这个边界，并使用双来源 token 校验，不回填或重写历史结果。

## 复现

没有下载模型或新依赖。本机已有 WSL 虚拟环境与 Qwen2.5-0.5B-Instruct 缓存。首次默认 V2 runner 在 sm_75 上报 FlashAttention 2 要求 sm_80+；改用 `VLLM_USE_V2_MODEL_RUNNER=0` 和 `TRITON_ATTN` 后仍因引擎 SIGBUS 启动失败。没有将这两次尝试写成部署成功，也没有为此升级驱动或修改环境。

另建 `reference_service.py`，复用同一缓存权重，在 Transformers eager 路径运行真实生成。这个参考服务串行处理生成，客户端并发请求会在服务内排队；它用于验证协议、token 指标、错误恢复和精度对照，**不是 vLLM 性能复现，也没有 continuous batching**。context 限制为 512；两种精度仅在对应实际成功证据生成后计为完成。

```bash
cd /mnt/c/Users/LW/Desktop/求职/04_作品与代码/vllm-sm75-throughput/catl_strengthening_20260918
/root/vllmbench/.venv/bin/python -m unittest -v test_benchmark.py
/root/vllmbench/.venv/bin/python audit_history.py
/root/vllmbench/.venv/bin/python -u run_local.py --backend transformers-reference
# 对已生成的实际目录交叉核对，再汇总（替换 RUN_DIRECTORY）：
/root/vllmbench/.venv/bin/python verify_reference.py --run RUN_DIRECTORY
/root/vllmbench/.venv/bin/python analyze_results.py
# 单独重试 vLLM 路径（本次两个已知启动失败已保留，勿当成功）：
# /root/vllmbench/.venv/bin/python -u run_local.py --backend vllm
```

`run_local.py` 仅监听 `127.0.0.1:8018`，若端口已占用就失败退出，不碰已有服务；强制 `HF_HUB_OFFLINE=1`。只清理它自己启动的进程组。每次独立创建 UTC 时间戳目录，旧证据不覆盖。若缺缓存或 GPU 不兼容，应保留真实失败，不用 replay/stub 假装成功。

评测其他已有服务（例如另行准备的 AWQ / GPTQ）时：

```bash
python quality_eval.py --url http://127.0.0.1:8018/v1/chat/completions \
  --model ACTUAL_SERVED_MODEL --label ACTUAL_VARIANT --output actual_variant_quality.json
```

`quality_eval.compare(paths)` 拒绝混用不同数据集哈希；相同提示仍须相同模型家族、版本、生成配置及任务定义才可归因到量化。16 道自建小题只是端到端冒烟检查，不能叫行业准确率、量化无损或模型能力验证。文本完全一致率也不等于任务准确率。

## 仍缺少的岗位证据

- 多 GPU / 分布式训练或推理、通信优化、异构集群资源调度尚未实做。
- 没有本轮 INT4 QAT / 新量化算法实现；旧 AWQ、GPTQ 是加载已有社区量化权重的对照。
- 没有本轮公开大规模任务基准、长上下文负载、持续生产流量、故障注入平台、正式 SLO 运维。
- 该练习为 AI 辅助实现，面试要能独立解释代码、指标与失败诊断；不能把工具产出直接等同于本人已熟练掌握。

## 文件

- `benchmark.py`：流式采集、双来源 token 计数、并发与异常恢复。
- `test_benchmark.py`：碎片化 SSE、UTF-8、多 token 事件、截断流、计数错误、失败吞吐、分位数边界。
- `quality_cases.json` / `quality_eval.py`：固定题目和答案、逐题原文与精确匹配。
- `verify_reference.py`：以相同参数调用独立 generate API，核对参考循环；历史默认重复惩罚不匹配的失败也保留。
- `run_local.py`：离线单卡运行、生命周期、环境和脚本哈希。
- `reference_service.py`：真实缓存 Qwen 模型的串行 Transformers 参考服务，区别于 vLLM 引擎。
- `audit_history.py`：旧结果复算，显式区分旧测量与新分析。
- `analyze_results.py`：验证完整原始结果后生成报告，未完成实验时拒绝生成性能报告。
- `inventory_model.py`：已缓存权重与配置哈希，不下载模型。
- `evidence/`：真实运行日志、原始请求、清理结果与复算数据。
