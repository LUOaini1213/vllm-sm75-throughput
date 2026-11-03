"""Streaming OpenAI-compatible benchmark with auditable token accounting.

No chunk-count-as-token assumption. vLLM return_token_ids plus final usage are
required for successful token metrics. All timings use a monotonic client clock.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import aiohttp


class SSEParser:
    """Incremental UTF-8 SSE framing, including comments/multiline data."""
    def __init__(self):
        self.buffer = b""
        self.lines = []

    def feed(self, chunk: bytes):
        self.buffer += chunk
        events = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line:
                if self.lines:
                    events.append(b"\n".join(self.lines).decode("utf-8"))
                self.lines = []
            elif line.startswith(b"data:"):
                self.lines.append(line[5:].lstrip(b" "))
        return events


def percentile(values, q):
    """Nearest-rank percentile; empty populations are null, never zero."""
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


def finalize(record):
    events = record["events"]
    token_events = [e for e in events if e["token_ids"]]
    content_events = [e for e in events if e["content"]]
    ids = [t for e in token_events for t in e["token_ids"]]
    record["token_ids"] = ids
    record["output_text"] = "".join(e["content"] for e in events)
    record["content_event_count"] = len(content_events)
    record["stream_token_count"] = len(ids)
    record["usage_completion_tokens"] = (record.get("usage") or {}).get("completion_tokens")
    record["token_count_matches_usage"] = record["usage_completion_tokens"] == len(ids)
    record["ttft_ms"] = token_events[0]["elapsed_s"] * 1000 if token_events else None
    record["ttfc_ms"] = content_events[0]["elapsed_s"] * 1000 if content_events else None
    record["ttft_including_client_queue_ms"] = (
        record["ttft_ms"] + record["client_queue_ms"] if token_events else None
    )
    # With >1 token in an SSE event, within-event token arrival times are unknown.
    # Do not silently report such intervals as exact per-token latency.
    single_token_events = bool(token_events) and all(len(e["token_ids"]) == 1 for e in token_events)
    record["single_token_events"] = single_token_events
    record["inter_token_arrival_ms"] = (
        [(b["elapsed_s"] - a["elapsed_s"]) * 1000
         for a, b in zip(token_events, token_events[1:])]
        if single_token_events else []
    )
    record["tpot_ms"] = (
        (token_events[-1]["elapsed_s"] - token_events[0]["elapsed_s"]) * 1000 / (len(ids) - 1)
        if single_token_events and len(ids) > 1 else None
    )
    if record["status"] == "pending":
        if not record["done"] or not record.get("finish_reason"):
            record["status"] = "incomplete_stream"
        elif not record["token_count_matches_usage"]:
            record["status"] = "token_accounting_error"
        elif not ids:
            record["status"] = "empty_output"
        else:
            record["status"] = "success"
    return record


async def request(session, url, body, *, request_id, queued_at=None, timeout=60, cancel_after=None):
    started = time.perf_counter()
    record = {"id": request_id, "client_queue_ms": (started - (queued_at or started)) * 1000,
              "body": body, "events": [], "status": "pending", "done": False,
              "http_status": None, "usage": None, "error": None}
    parser = SSEParser()
    try:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            record["http_status"] = response.status
            if response.status != 200:
                record["status"] = "http_error"
                record["error"] = (await response.text())[:4096]
            else:
                stop = False
                async for chunk in response.content.iter_any():
                    arrived = time.perf_counter() - started
                    for payload in parser.feed(chunk):
                        if payload == "[DONE]":
                            record["done"] = True
                            continue
                        obj = json.loads(payload)
                        if "error" in obj:
                            record["status"] = "server_error"
                            record["error"] = obj["error"]
                            stop = True
                            break
                        if obj.get("usage"):
                            record["usage"] = obj["usage"]
                        for choice in obj.get("choices", []):
                            record["events"].append({"elapsed_s": arrived,
                                "token_ids": choice.get("token_ids") or [],
                                "content": (choice.get("delta") or {}).get("content") or ""})
                            if choice.get("finish_reason"):
                                record["finish_reason"] = choice["finish_reason"]
                        if cancel_after and sum(len(e["token_ids"]) for e in record["events"]) >= cancel_after:
                            record["status"] = "client_cancelled"
                            stop = True
                            break
                    if stop:
                        break
    except asyncio.TimeoutError:
        record["status"] = "client_timeout"
    except (aiohttp.ClientError, ValueError, UnicodeDecodeError) as exc:
        record["status"] = "protocol_or_connection_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["request_wall_s"] = time.perf_counter() - started
    return finalize(record)


def summarize(records, wall_s, *, ttft_slo_ms=1000):
    successful = [r for r in records if r["status"] == "success"]
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    good = [r for r in successful if r["ttft_ms"] <= ttft_slo_ms]
    return {"requests": len(records), "statuses": counts, "wall_s": wall_s,
            "success_rate": len(successful) / len(records) if records else None,
            "output_tokens": sum(r["stream_token_count"] for r in successful),
            "successful_tokens_per_s": sum(r["stream_token_count"] for r in successful) / wall_s,
            "successful_requests_per_s": len(successful) / wall_s,
            "ttft_slo_ms": ttft_slo_ms, "slo_successful_requests": len(good),
            "slo_goodput_requests_per_s": len(good) / wall_s,
            "ttft_p50_ms": percentile([r["ttft_ms"] for r in successful], .5),
            "ttft_p95_ms": percentile([r["ttft_ms"] for r in successful], .95),
            "client_queue_p95_ms": percentile([r["client_queue_ms"] for r in records], .95),
            "tpot_p50_ms": percentile([r["tpot_ms"] for r in successful if r["tpot_ms"] is not None], .5),
            "itl_p95_ms": percentile([t for r in successful for t in r["inter_token_arrival_ms"]], .95),
            "multi_token_event_requests": sum(not r["single_token_events"] for r in successful),
            "content_event_count_equals_tokens": sum(r["content_event_count"] == r["stream_token_count"] for r in successful)}


def body(model, prompt, max_tokens=32, **extra):
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 42,
            "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True, **extra}


async def run_sweep(url, model, output, repeats=2):
    batches = []
    async with aiohttp.ClientSession() as session:
        warmup = await request(session, url, body(model, "Count from one to four.", 8), request_id="warmup")
        if warmup["status"] != "success":
            raise RuntimeError(f"Warmup failed: {warmup}")
        for repeat in range(repeats):
            for concurrency in [1, 2, 4]:
                semaphore = asyncio.Semaphore(concurrency)
                start = time.perf_counter()
                async def guarded(index):
                    queued = time.perf_counter()
                    async with semaphore:
                        return await request(session, url,
                            body(model, f"Explain distributed database concept number {index} in detail.", 32, ignore_eos=True),
                            request_id=f"r{repeat}-c{concurrency}-{index}", queued_at=queued)
                records = await asyncio.gather(*(guarded(i) for i in range(8)))
                summary = summarize(records, time.perf_counter() - start)
                batches.append({"repeat": repeat, "concurrency": concurrency, "summary": summary, "records": records})
                print(json.dumps({"repeat": repeat, "concurrency": concurrency, **summary}), flush=True)
    result = {"schema": 1, "model": model, "url": url, "kind": "real_service_closed_loop",
              "warmup": warmup, "batches": batches}
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


async def run_reliability(url, model, output):
    cases = []
    async with aiohttp.ClientSession() as session:
        definitions = [
            ("unknown_model", body("definitely-not-a-model", "Hello"), {}, "http_error", 404),
            ("context_overflow", body(model, "data " * 1500), {}, "http_error", 400),
            ("invalid_max_tokens", body(model, "Hello", -1), {}, "http_error", 400),
            ("client_timeout", body(model, "Write a detailed essay.", 128), {"timeout": .001}, "client_timeout", None),
            ("client_cancel", body(model, "Write a detailed essay.", 128, ignore_eos=True), {"cancel_after": 3}, "client_cancelled", 200),
        ]
        for name, payload, options, expected, status_code in definitions:
            observed = await request(session, url, payload, request_id=name, **options)
            recovery = await request(session, url, body(model, "Reply with OK only.", 8), request_id=name + "-recovery")
            cases.append({"case": name, "expected": expected, "record": observed, "recovery": recovery,
                          "passed": observed["status"] == expected and
                              (status_code is None or observed["http_status"] == status_code) and recovery["status"] == "success"})
    result = {"kind": "real_service_fault_probes", "cases": cases,
              "passed": sum(c["passed"] for c in cases), "total": len(cases)}
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8018/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reliability", action="store_true")
    args = parser.parse_args()
    asyncio.run((run_reliability if args.reliability else run_sweep)(args.url, args.model, args.output))
