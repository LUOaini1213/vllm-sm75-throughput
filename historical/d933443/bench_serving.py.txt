# -*- coding: utf-8 -*-
"""vLLM 服务端压测：并发扫描下的吞吐 / TTFT / TPOT，以及前缀复用的效果。

指标定义（与 JD 里那套一致）：
  TTFT  Time To First Token   —— 首 token 延迟，决定交互体感
  TPOT  Time Per Output Token —— 首 token 之后的平均出词间隔
  吞吐   总输出 token 数 / 墙钟时间

用法（先 bash serve.sh 起服务）：
  python bench_serving.py                 # 并发扫描
  python bench_serving.py --prefix-test   # 前缀复用对照
"""
import argparse, asyncio, json, statistics as st, time
import aiohttp

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
    t0 = time.perf_counter()
    ttft, n_tok, last = None, 0, t0
    async with sess.post(URL, json=body) as resp:
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            if delta.get("content"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                n_tok += 1
                last = now
    return dict(ttft=ttft or 0.0, total=last - t0, n_tok=n_tok)


async def run_batch(n_conc, n_req, max_tokens, use_prefix):
    prompts = [f"Explain concept #{i} in distributed systems." for i in range(n_req)]
    sem = asyncio.Semaphore(n_conc)

    async def guarded(sess, p):
        async with sem:
            return await one_request(sess, p, max_tokens, use_prefix)

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        await one_request(sess, "warmup", 4, use_prefix)          # 预热
        t0 = time.perf_counter()
        rs = await asyncio.gather(*(guarded(sess, p) for p in prompts))
        wall = time.perf_counter() - t0

    tot_tok = sum(r["n_tok"] for r in rs)
    tpots = [(r["total"] - r["ttft"]) / max(r["n_tok"] - 1, 1) for r in rs if r["n_tok"] > 1]
    return dict(conc=n_conc, wall=wall, tput=tot_tok / wall, rps=len(rs) / wall,
                ttft_p50=st.median(r["ttft"] for r in rs),
                ttft_p99=sorted(r["ttft"] for r in rs)[int(len(rs) * 0.99) - 1],
                tpot_p50=st.median(tpots) if tpots else 0.0, tot_tok=tot_tok)


async def sweep(args):
    print(f"{'并发':>5}{'请求':>6}{'墙钟s':>9}{'吞吐 tok/s':>13}{'RPS':>8}"
          f"{'TTFT p50':>11}{'TTFT p99':>11}{'TPOT p50':>11}")
    print("-" * 74)
    out = []
    for c in [1, 2, 4, 8, 16, 32]:
        r = await run_batch(c, max(c * 4, 16), args.max_tokens, use_prefix=False)
        print(f"{r['conc']:>5}{max(c*4,16):>6}{r['wall']:>9.2f}{r['tput']:>13.1f}"
              f"{r['rps']:>8.2f}{r['ttft_p50']*1e3:>10.1f}ms{r['ttft_p99']*1e3:>10.1f}ms"
              f"{r['tpot_p50']*1e3:>10.2f}ms")
        out.append(r)
    json.dump(out, open("sweep_results.json", "w"), indent=1)
    base = out[0]["tput"]
    print(f"\ncontinuous batching 收益：并发 1 → 32，吞吐 "
          f"{base:.1f} → {out[-1]['tput']:.1f} tok/s（{out[-1]['tput']/base:.1f}×），"
          f"TTFT p50 {out[0]['ttft_p50']*1e3:.0f} → {out[-1]['ttft_p50']*1e3:.0f} ms")
    print("吞吐与延迟的取舍就在这张表里：并发拉高吞吐涨，但 TTFT 同步恶化。")


async def prefix_test(args):
    print("前缀复用对照（服务端需带 --enable-prefix-caching 启动）")
    print(f"{'场景':<26}{'吞吐 tok/s':>13}{'TTFT p50':>12}")
    print("-" * 51)
    for label, up in [("无共享前缀", False), (f"共享前缀 ({len(SHARED_PREFIX)} 字符)", True)]:
        r = await run_batch(8, 32, args.max_tokens, use_prefix=up)
        print(f"{label:<26}{r['tput']:>13.1f}{r['ttft_p50']*1e3:>11.1f}ms")
    print("\n共享前缀命中 KV cache 后，重复的 prefill 不再重算，TTFT 应显著下降。")
    print("对比未开 --enable-prefix-caching 重启服务再跑一次，差值即为该特性的真实收益。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prefix-test", action="store_true")
    a = ap.parse_args()
    asyncio.run(prefix_test(a) if a.prefix_test else sweep(a))
