#!/usr/bin/env bash
# 对照组：--enforce-eager 关闭 CUDA Graph 与 torch.compile，其余参数与 serve.sh 完全一致。
cd "$HOME/vllmbench"
source .venv/bin/activate
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --dtype half \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --enforce-eager \
    --port 8000
