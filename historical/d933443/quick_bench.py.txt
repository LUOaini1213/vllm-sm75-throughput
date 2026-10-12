# -*- coding: utf-8 -*-
"""定点测量：固定并发下的 TPOT，用于对比不同服务端配置。"""
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

async def run(conc, n_req, n_tok):
    sem=asyncio.Semaphore(conc)
    async def g(s,i):
        async with sem: return await one(s,i,n_tok)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        await one(s,-1,4)
        t0=time.perf_counter()
        rs=await asyncio.gather(*(g(s,i) for i in range(n_req)))
        wall=time.perf_counter()-t0
    tp=[(tot-tt)/max(n-1,1) for tt,tot,n in rs if n>1]
    tot_tok=sum(n for _,_,n in rs)
    return wall, tot_tok/wall, st.median(tp)*1e3

if __name__=="__main__":
    conc=int(sys.argv[1]) if len(sys.argv)>1 else 8
    wall,tput,tpot=asyncio.run(run(conc, conc*3, 64))
    print(f"并发={conc}  墙钟={wall:.1f}s  吞吐={tput:.1f} tok/s  TPOT p50={tpot:.2f} ms")
