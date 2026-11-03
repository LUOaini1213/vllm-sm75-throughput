# -*- coding: utf-8 -*-
"""隔离测试：一次性发 N 条并发请求，期间无新请求到达。
用于区分「稳态 batch=N 解码慢」与「新请求 prefill 插队拖累解码」。

2026-11-03 起改用与 bench_serving.py / quick_bench.py 相同的 serving_metrics 客户端：
严格 HTTP/SSE 校验，token 数只取服务端最终 usage.completion_tokens，
任何一条失败或缺 usage 时该档吞吐记为不可用，退出码 2。
README 第三、五节那两张突发表是旧版脚本测的（按非空内容片段计数、不查状态码），
旧版原始字节冻结在 historical/d933443/isolate.py.txt，本脚本不能复现或替代那两张表。

用法（先 bash serve.sh 起服务）：python isolate.py   结果另写 isolate_results_v2.json
"""
import asyncio
import json
import sys
import time

import aiohttp
from serving_metrics import client_identity, measure_request, summarize_requests

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BURSTS = [1, 2, 3, 4, 8, 16]


async def one(session, i, n_tok):
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"Explain idea {i} briefly."}],
            "max_tokens": n_tok, "temperature": 0.0}
    return await measure_request(session, URL, body)


async def burst(n, n_tok=64):
    """严格同时发 n 条，全部跑完才结束 —— 稳态就是 batch=n。"""
    if min(n, n_tok) <= 0:
        raise ValueError("burst size and max_tokens must be positive")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        warmup = await one(session, -1, 4)
        if not warmup["ok"]:
            raise RuntimeError("warmup_failed: " + warmup["error"])
        started = time.perf_counter()
        results = await asyncio.gather(*(one(session, i, n_tok) for i in range(n)))
        wall = time.perf_counter() - started
    return dict(summarize_requests(results, wall), burst=n, model=MODEL, max_tokens=n_tok,
                client_source=client_identity(__file__))


def metric(value, *, scale=1, digits=1):
    return "N/A" if value is None else f"{value * scale:.{digits}f}"


async def sweep():
    print("=== 无新到达的纯稳态测试 ===")
    out = []
    for n in BURSTS:
        r = await burst(n)
        print(f"  一次性 {n:>2} 条并发: {r['successful']}/{r['attempted']} 成功  墙钟 {r['wall']:>6.2f}s  "
              f"吞吐 {metric(r['tput']):>6} tok/s  TPOT估计 p50 {metric(r['tpot_estimate_p50'], scale=1000, digits=2):>7} ms  "
              f"{r['measurement_status']} errors={json.dumps(r['errors'], sort_keys=True)}")
        out.append(r)
    with open("isolate_results_v2.json", "w", encoding="utf-8") as target:
        json.dump(out, target, indent=1, allow_nan=False)
    if any(r["measurement_status"] != "complete" for r in out):
        print("测量不完整：吞吐不可用。失败与缺失 usage 已写入 isolate_results_v2.json。")
        return 2
    return 0


def main():
    try:
        return asyncio.run(sweep())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
