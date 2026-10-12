# -*- coding: utf-8 -*-
"""定点测量：HTTP/SSE 完整性及服务端 token 用量；无逐 token 时间证明。"""
import argparse
import asyncio
import json
import sys
import time
import aiohttp
from serving_metrics import client_identity, measure_request, summarize_requests

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


async def one(session, i, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"Explain idea {i} briefly."}],
            "max_tokens": max_tokens, "temperature": 0.0}
    return await measure_request(session, URL, body)


async def run(conc, n_req, max_tokens):
    if min(conc, n_req, max_tokens) <= 0:
        raise ValueError("concurrency, request count and max_tokens must be positive")
    sem = asyncio.Semaphore(conc)

    async def guarded(session, i):
        async with sem:
            return await one(session, i, max_tokens)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        warmup = await one(session, -1, 4)
        if not warmup["ok"]:
            raise RuntimeError("warmup_failed: " + warmup["error"])
        started = time.perf_counter()
        results = await asyncio.gather(*(guarded(session, i) for i in range(n_req)))
        wall = time.perf_counter() - started
    return dict(summarize_requests(results, wall), conc=conc, model=MODEL, max_tokens=max_tokens,
                client_source=client_identity(__file__))


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("concurrency", nargs="?", type=positive_int, default=8)
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args.concurrency, args.concurrency * 3, 64))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["measurement_status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
