# -*- coding: utf-8 -*-
"""vLLM 服务端压测：并发扫描下的吞吐 / TTFT / TPOT，以及前缀复用的效果。

指标边界：TTFT 是首个非空内容事件延迟；TPOT 是事件时间估计。
吞吐使用服务端最终 usage.completion_tokens，缺失时不可用。

用法（先 bash serve.sh 起服务）：
  python bench_serving.py                 # 并发扫描
  python bench_serving.py --prefix-test   # 前缀复用对照
"""
import argparse, asyncio, json, sys, time
import aiohttp
from serving_metrics import client_identity, measure_request, summarize_requests

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# 一段较长的共享 system prompt：开 --enable-prefix-caching 后其 prefill 只算一次
SHARED_PREFIX = (
    "You are a meticulous technical assistant. Answer concisely and precisely. "
    "Always reason step by step before answering. " * 20
)


async def one_request(sess, prompt, max_tokens, use_prefix):
    msgs = ([{"role": "system", "content": SHARED_PREFIX}] if use_prefix else []) + \
           [{"role": "user", "content": prompt}]
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True}
    return await measure_request(sess, URL, body)


async def run_batch(n_conc, n_req, max_tokens, use_prefix):
    if min(n_conc, n_req, max_tokens) <= 0:
        raise ValueError("concurrency, request count and max_tokens must be positive")
    prompts = [f"Explain concept #{i} in distributed systems." for i in range(n_req)]
    sem = asyncio.Semaphore(n_conc)

    async def guarded(sess, p):
        async with sem:
            return await one_request(sess, p, max_tokens, use_prefix)

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        warmup = await one_request(sess, "warmup", 4, use_prefix)
        if not warmup["ok"]:
            raise RuntimeError("warmup_failed: " + warmup["error"])
        t0 = time.perf_counter()
        rs = await asyncio.gather(*(guarded(sess, p) for p in prompts))
        wall = time.perf_counter() - t0

    return dict(summarize_requests(rs, wall), conc=n_conc, model=MODEL, max_tokens=max_tokens,
                shared_prefix=use_prefix, client_source=client_identity(__file__))


def metric(value, *, scale=1, digits=1):
    return "N/A" if value is None else f"{value * scale:.{digits}f}"


async def sweep(args):
    print("并发 成功/请求 墙钟s 吞吐tok/s 成功RPS TTFT内容p50/p99(ms) TPOT估计p50(ms) 状态")
    out = []
    for c in [1, 2, 4, 8, 16, 32]:
        r = await run_batch(c, max(c * 4, 16), args.max_tokens, use_prefix=False)
        print(f"{c} {r['successful']}/{r['attempted']} {r['wall']:.2f} {metric(r['tput'])} "
              f"{r['rps_success']:.2f} {metric(r['ttft_p50'], scale=1000)}/"
              f"{metric(r['ttft_p99'], scale=1000)} {metric(r['tpot_estimate_p50'], scale=1000)} "
              f"{r['measurement_status']} failed={r['failed']} missing_usage={r['tokens_unavailable']} "
              f"errors={json.dumps(r['errors'], sort_keys=True)}")
        out.append(r)
    with open("sweep_results_v2.json", "w", encoding="utf-8") as target:
        json.dump(out, target, indent=1, allow_nan=False)
    if any(r["measurement_status"] != "complete" for r in out):
        print("测量不完整：吞吐/收益不可用。失败与缺失 usage 已写入 sweep_results_v2.json。")
        return 2
    base = out[0]["tput"]
    if base > 0:
        print(f"\n实测吞吐比（并发 32 / 1）：{out[-1]['tput']/base:.1f}×")
    else:
        print("\n基线输出 token 为零，吞吐比不可用。")
    return 0


async def prefix_test(args):
    print("前缀复用对照（服务端需带 --enable-prefix-caching 启动）")
    print(f"{'场景':<26}{'吞吐 tok/s':>13}{'TTFT内容 p50':>16}")
    print("-" * 51)
    out = []
    for label, up in [("无共享前缀", False), (f"共享前缀 ({len(SHARED_PREFIX)} 字符)", True)]:
        r = await run_batch(8, 32, args.max_tokens, use_prefix=up)
        out.append(dict(r, shared_prefix=up))
        print(f"{label:<26}{metric(r['tput']):>13}{metric(r['ttft_p50'], scale=1000):>15}ms "
              f"{r['measurement_status']} failed={r['failed']} missing_usage={r['tokens_unavailable']} "
              f"errors={json.dumps(r['errors'], sort_keys=True)}")
    with open("prefix_results_v2.json", "w", encoding="utf-8") as target:
        json.dump(out, target, indent=1, allow_nan=False)
    print("\n此处改变输入前缀；因果对照仍需同输入、独立控制服务端 prefix-caching 开/关。")
    return 0 if all(r["measurement_status"] == "complete" for r in out) else 2


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=positive_int, default=128)
    ap.add_argument("--prefix-test", action="store_true")
    args = ap.parse_args()
    try:
        return asyncio.run(prefix_test(args) if args.prefix_test else sweep(args))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
