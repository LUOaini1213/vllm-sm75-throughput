# -*- coding: utf-8 -*-
"""隔离测试：一次性发 N 条并发请求，期间无新请求到达。
用于区分「稳态 batch=N 解码慢」与「新请求 prefill 插队拖累解码」。"""
import asyncio, sys, time, json, statistics as st
import aiohttp
URL="http://127.0.0.1:8000/v1/chat/completions"; MODEL="Qwen/Qwen2.5-0.5B-Instruct"

async def one(s, i, n_tok):
    b={"model":MODEL,"messages":[{"role":"user","content":f"Explain idea {i} briefly."}],
       "max_tokens":n_tok,"temperature":0.0,"stream":True}
    t0=time.perf_counter(); ttft=None; n=0; last=t0
    async with s.post(URL,json=b) as r:
        async for raw in r.content:
            l=raw.decode().strip()
            if not l.startswith("data: ") or l=="data: [DONE]": continue
            d=json.loads(l[6:])["choices"][0].get("delta",{})
            if d.get("content"):
                now=time.perf_counter()
                if ttft is None: ttft=now-t0
                n+=1; last=now
    return (ttft or 0), last-t0, n

async def burst(n, n_tok=64):
    """严格同时发 n 条，全部跑完才结束 —— 稳态就是 batch=n。"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        await one(s,-1,4)                       # 预热
        t0=time.perf_counter()
        rs=await asyncio.gather(*(one(s,i,n_tok) for i in range(n)))
        wall=time.perf_counter()-t0
    tp=[(tot-tt)/max(k-1,1) for tt,tot,k in rs if k>1]
    tot=sum(k for _,_,k in rs)
    print(f"  一次性 {n:>2} 条并发: 墙钟 {wall:>6.2f}s  总吞吐 {tot/wall:>6.1f} tok/s  "
          f"TPOT p50 {st.median(tp)*1e3:>7.2f} ms")

async def main():
    print("=== 无新到达的纯稳态测试 ===")
    for n in [1, 2, 3, 4, 8, 16]:
        await burst(n)

asyncio.run(main())
