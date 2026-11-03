# 兼容性与检查记录

1. `local_20260918T053721Z`：vLLM 0.28.0 默认 V2 runner 启动失败。服务日志出现 `Cannot use FA version 2 ... compute capability >= 8`，之后 Engine core initialization failed。没有把日志中的一条告警直接认定为整个崩溃的唯一原因。
2. `local_20260918T053934Z`：显式 `VLLM_USE_V2_MODEL_RUNNER=0`、`TRITON_ATTN`、eager 后仍启动失败。引擎日志最后在分布式单卡初始化附近终止。排查时终端 `dmesg` 输出出现以下片段（下文是当时工具输出的摘录，不是完整原始内核日志文件）：

   ```text
   [  173.600151] python3.12: VLLM::EngineCor: potentially unexpected fatal signal 7.
   [  173.601561] WSL (2678 - CaptureCrash): Capturing crash for pid: 659, executable: !usr!bin!python3.12, signal: 7, port: 50005
   ```

   两次启动失败均保留 manifest 与完整服务日志，没有修改驱动、安装新包或继续扩大排障范围。
3. 直接 CUDA 张量矩阵乘成功；同一缓存权重随后可由 Transformers eager 在本地 GPU 运行。这只缩小故障范围，不足以认定 vLLM 具体 bug。
4. Transformers 参考服务初次将新版 `apply_chat_template` 默认返回对象直接转张量，报 `Could not infer dtype of tokenizers.Encoding`。改为显式 `return_dict=False`，再以新时间戳目录重新实跑，失败目录未覆盖。
5. 独立 `model.generate` 初次核对为 2/4 完全一致。核对发现缓存 `generation_config.json` 自带 `repetition_penalty=1.1`，而手写参考循环为原始 argmax、等价于 1.0。将独立路径显式设为 1.0 后另存 `generate_equivalence_repetition1.json`，原失败文件 `generate_equivalence.json` 保留。未修改题目、答案或主评分规则。

本目录包含一次成功的真实 GPU 参考服务实验，不包含新的 vLLM 成功压测、多卡结果或 INT4 量化质量结果。
