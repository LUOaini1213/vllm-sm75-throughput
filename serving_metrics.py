"""Client-observed streaming metrics; no tokenizer or model dependency.

Only a terminal server usage record supplies completion_tokens. Content events
are diagnostics, not tokens. This client handles n=1 text chat completions.
"""
import json
import hashlib
import math
import statistics
import time
from collections import Counter
from pathlib import Path


class StreamFailure(Exception):
    """A safe diagnostic code, never response text or exception details."""


def client_identity(entrypoint):
    """Record revised client bytes without claiming any historical replay."""
    paths = [Path(entrypoint).resolve(), Path(__file__).resolve()]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class SSEFrames:
    """Incremental UTF-8 SSE framing, independent of network chunk boundaries."""
    def __init__(self):
        self.buffer = b""
        self.data = []
        self.event = ""
        self.size = 0
        self.first_line = True

    def feed(self, chunk=b"", *, eof=False):
        self.buffer += chunk
        frames = []
        while True:
            positions = [p for p in (self.buffer.find(b"\r"), self.buffer.find(b"\n")) if p >= 0]
            if not positions:
                break
            p = min(positions)
            if self.buffer[p:p + 1] == b"\r" and p == len(self.buffer) - 1 and not eof:
                break
            step = 2 if self.buffer[p:p + 2] == b"\r\n" else 1
            raw, self.buffer = self.buffer[:p], self.buffer[p + step:]
            self.size += len(raw)
            if self.size > 1024 * 1024:
                raise StreamFailure("frame_too_large")
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise StreamFailure("invalid_utf8") from None
            if self.first_line:
                line = line.removeprefix("\ufeff")
                self.first_line = False
            if not line:
                if self.data or self.event == "error":
                    frames.append((self.event, "\n".join(self.data)))
                self.data, self.event, self.size = [], "", 0
            elif not line.startswith(":"):
                field, sep, value = line.partition(":")
                value = value.removeprefix(" ") if sep else ""
                if field == "data":
                    self.data.append(value)
                elif field == "event":
                    self.event = value
        if len(self.buffer) + self.size > 1024 * 1024:
            raise StreamFailure("frame_too_large")
        if eof and (self.buffer or self.data or self.event):
            raise StreamFailure("truncated_frame")
        return frames


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StreamFailure("duplicate_json_key")
        result[key] = value
    return result


async def consume_stream(content, started, *, clock=time.perf_counter):
    frames = SSEFrames()
    done = finished = False
    ttft = last = tokens = None
    events = 0

    def accept(event, data):
        nonlocal done, finished, ttft, last, tokens, events
        if done:
            raise StreamFailure("data_after_done")
        if event == "error":
            raise StreamFailure("server_error")
        if data == "[DONE]":
            if not finished:
                raise StreamFailure("missing_finish_reason")
            done = True
            return
        try:
            obj = json.loads(data, object_pairs_hook=_object)
        except (ValueError, TypeError):
            raise StreamFailure("invalid_json") from None
        if not isinstance(obj, dict):
            raise StreamFailure("invalid_chunk")
        if "error" in obj:
            raise StreamFailure("server_error")
        choices = obj.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamFailure("invalid_choices")
        usage = obj.get("usage")
        if not choices:
            if not finished or not isinstance(usage, dict) or tokens is not None:
                raise StreamFailure("invalid_final_usage")
            count = usage.get("completion_tokens")
            if type(count) is not int or count < 0 or (events and count == 0):
                raise StreamFailure("invalid_completion_tokens")
            tokens = count
            return
        # Requested final-only usage: never trust interim counts as final totals.
        if usage is not None:
            raise StreamFailure("nonterminal_usage")
        if finished or tokens is not None:
            raise StreamFailure("choice_after_finish")
        choice = choices[0]
        if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0:
            raise StreamFailure("invalid_choice")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise StreamFailure("invalid_delta")
        text = delta.get("content")
        if text is not None and not isinstance(text, str):
            raise StreamFailure("invalid_content")
        if delta.get("role") not in (None, "assistant") or any(
                key not in ("role", "content") and value is not None and value != "" and value != []
                for key, value in delta.items()):
            raise StreamFailure("unsupported_delta")
        if text:
            now = clock() - started
            ttft = now if ttft is None else ttft
            last = now
            events += 1
        reason = choice.get("finish_reason")
        if reason is not None:
            if reason not in ("stop", "length"):
                raise StreamFailure("unsupported_finish_reason")
            finished = True

    async for chunk in content:
        if not isinstance(chunk, bytes):
            raise StreamFailure("invalid_transport_chunk")
        for event, data in frames.feed(chunk):
            accept(event, data)
    for event, data in frames.feed(eof=True):
        accept(event, data)
    if not done:
        raise StreamFailure("missing_done")
    return {"ok": True, "error": None, "ttft": ttft,
            "last_content": last, "completion_tokens": tokens,
            "token_source": "server_final_usage" if tokens is not None else "unavailable",
            "content_events": events}


async def measure_request(session, url, body):
    started = time.perf_counter()
    request = dict(body, stream=True, n=1, stream_options={"include_usage": True})
    status = None
    try:
        async with session.post(url, json=request) as response:
            status = response.status
            if not 200 <= status < 300:
                raise StreamFailure("http_error")
            media = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if media != "text/event-stream":
                raise StreamFailure("invalid_content_type")
            result = await consume_stream(response.content, started)
    except StreamFailure as exc:
        result = {"ok": False, "error": str(exc)}
    except Exception:
        # Includes timeout/disconnect/aiohttp payload errors. Do not expose URLs,
        # response bodies, prompts, or arbitrary exception text in diagnostics.
        result = {"ok": False, "error": "transport_error"}
    result.update(http_status=status, elapsed=time.perf_counter() - started)
    return result


def summarize_requests(results, wall):
    if not results or not math.isfinite(wall) or wall <= 0:
        raise ValueError("nonempty results and a positive finite wall time are required")
    successful = [r for r in results if r["ok"]]
    failed = len(results) - len(successful)
    missing = sum(r["completion_tokens"] is None for r in successful)
    complete = not failed and not missing
    ttfts = sorted(r["ttft"] for r in successful if r["ttft"] is not None)
    # This is an event-timed estimate, not per-token server decode latency.
    tpots = [(r["last_content"] - r["ttft"]) / (r["completion_tokens"] - 1)
             for r in successful if r["completion_tokens"] is not None
             and r["completion_tokens"] > 1 and r["content_events"] > 1]
    total = sum(r["completion_tokens"] for r in successful) if complete else None
    return {"schema_version": 2, "measurement_status": "complete" if complete else "incomplete",
            "attempted": len(results), "successful": len(successful), "failed": failed,
            "tokens_unavailable": missing, "errors": dict(Counter(r["error"] for r in results if not r["ok"])),
            "wall": wall, "rps_success": len(successful) / wall,
            "rps_attempted": len(results) / wall,
            "tot_tok": total, "tput": total / wall if complete else None,
            "ttft_p50": statistics.median(ttfts) if ttfts else None,
            "ttft_p99": ttfts[math.ceil(len(ttfts) * .99) - 1] if ttfts else None,
            "ttft_observations": len(ttfts),
            "tpot_estimate_p50": statistics.median(tpots) if tpots else None,
            "tpot_estimate_observations": len(tpots)}
