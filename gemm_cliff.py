# -*- coding: utf-8 -*-
"""脱离 vLLM 验证：sm_75 无 Tensor Core 时，fp16 GEMM 在 M=1→2 是否发生性能塌陷。

形状取自 Qwen2.5-0.5B：hidden=896, intermediate=4864。
解码时每步的核心运算就是 [M, 896] @ [896, N]，M = 并发序列数。
"""
import torch, statistics as st

H, I = 896, 4864
SHAPES = [("MLP gate/up  [M,896]@[896,4864]", H, I),
          ("MLP down     [M,4864]@[4864,896]", I, H),
          ("QKV proj     [M,896]@[896,1152]", H, 1152)]

def bench(M, K, N, dtype, iters=200):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    for _ in range(20): a @ b
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); a @ b; e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    t = st.median(ts)
    # 权重矩阵是主要流量：K*N 个元素
    gbs = K * N * a.element_size() / t * 1e3 / 2**30
    return t, gbs

p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} | sm_{p.major}{p.minor} | torch {torch.__version__}")
print("注：GTX 16 系为 Turing 但移除了 Tensor Core，fp16 无张量核加速。\n")

for dtype, tag in [(torch.float16, "fp16"), (torch.float32, "fp32")]:
    print(f"########## {tag} ##########")
    for name, K, N in SHAPES:
        print(f"\n{name}")
        print(f"  {'M':>4}{'耗时 ms':>11}{'权重带宽 GiB/s':>17}{'相对 M=1':>11}")
        base = None
        for M in [1, 2, 4, 8, 16, 32, 64]:
            t, gbs = bench(M, K, N, dtype)
            if base is None: base = gbs
            print(f"  {M:>4}{t:>11.4f}{gbs:>17.1f}{gbs/base:>10.2f}×")
