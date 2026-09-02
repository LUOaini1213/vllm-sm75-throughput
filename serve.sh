#!/usr/bin/env bash
# 启动 vLLM OpenAI 兼容服务。针对 sm_75 + 4 GiB 的关键参数都在这里。
cd "$HOME/vllmbench"
source .venv/bin/activate

# WSL2 专有：vLLM 默认在 WSL 下关闭锁页内存(pin_memory)，而 v0.28 的 UvaBuffer
# 硬依赖它，不开就直接 "RuntimeError: UVA is not available" 起不来。
# 本机内核 6.18.33 >= vLLM 要求的 4.19.121，锁页内存实际可用，显式打开即可。
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# --dtype half             : Turing(sm_75) 无 bf16 硬件支持，必须 fp16
# --max-model-len          : 直接决定 KV cache 大小；4 GiB 卡上最敏感的旋钮
# --gpu-memory-utilization : 4 GiB 卡上 Windows 桌面已占约 0.78 GiB，
#                            实际可用仅 3.22 GiB，设 0.85 会直接启动失败
# --enable-prefix-caching  : 前缀复用，相同 system prompt 的 prefill 只算一次
# --max-num-seqs           : continuous batching 并发上限
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --dtype half \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --port 8000
