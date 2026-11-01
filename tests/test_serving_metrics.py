"""Offline wire-byte fixtures. Fake transport does NOT validate a real server."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import bench_serving
import quick_bench
from serving_metrics import consume_stream, measure_request, summarize_requests, StreamFailure

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_bytes()


def event(obj):
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()


def choice(text="", finish=None):
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


def usage(tokens):
    return {"choices": [], "usage": {"completion_tokens": tokens}}


class Content:
    def __init__(self, chunks, error=None):
        self.chunks, self.error = chunks, error
        self.iterated = False

    def __aiter__(self):
        return self.iterate()

    async def iterate(self):
        self.iterated = True
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


class Response:
    def __init__(self, wire=b"", status=200, media="text/event-stream", error=None):
        self.status = status
        self.headers = {"Content-Type": media}
        self.content = Content([wire], error)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def post(self, url, json):
        self.requests.append(json)
        return next(self.responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class ParserTests(unittest.IsolatedAsyncioTestCase):
    async def parse(self, wire, chunks=None, clock=None):
        return await consume_stream(Content(chunks or [wire]), 0, clock=clock or (lambda: 1))

    async def reject(self, wire, expected, **kwargs):
        with self.assertRaisesRegex(StreamFailure, "^" + expected + "$"):
            await self.parse(wire, **kwargs)

    async def test_normal_terminal_usage_not_content_events(self):
        result = await self.parse(fixture("normal.sse"), clock=iter([1, 3]).__next__)
        self.assertEqual(result["completion_tokens"], 7)
        self.assertEqual(result["content_events"], 2)
        self.assertEqual(result["token_source"], "server_final_usage")
        self.assertEqual(result["ttft"], 1)
        self.assertEqual(result["last_content"], 3)

    async def test_usage_only_terminal_chunk_and_length_finish(self):
        result = await self.parse(event(choice("Text", "length")) + event(usage(3)) + b"data: [DONE]\n\n")
        self.assertEqual(result["completion_tokens"], 3)

    async def test_missing_usage_is_unavailable(self):
        result = await self.parse(fixture("missing_usage.sse"))
        self.assertTrue(result["ok"])
        self.assertIsNone(result["completion_tokens"])
        self.assertEqual(result["token_source"], "unavailable")

    async def test_missing_done_is_failure(self):
        await self.reject(fixture("no_done.sse"), "missing_done")

    async def test_error_chunk_is_failure_even_with_done(self):
        await self.reject(fixture("server_error.sse"), "server_error")

    async def test_explicit_error_event(self):
        await self.reject(b"event: error\ndata: {}\n\n", "server_error")

    async def test_partial_frame_at_eof(self):
        await self.reject(fixture("normal.sse").rstrip(b"\n") + b"\n", "truncated_frame")

    async def test_finish_required_not_done_alone(self):
        await self.reject(b"data: [DONE]\n\n", "missing_finish_reason")

    async def test_json_and_utf8_rejected(self):
        await self.reject(b"data: {broken}\n\n", "invalid_json")
        await self.reject(b"data: \xff\n\n", "invalid_utf8")

    async def test_arbitrary_utf8_and_crlf_chunk_splits(self):
        wire = b"\xef\xbb\xbf" + event(choice("你好", "stop")) + event(usage(2)) + b"data: [DONE]\n\n"
        for ending in [b"\n", b"\r\n", b"\r"]:
            changed = wire.replace(b"\n", ending)
            result = await self.parse(changed, chunks=[changed[i:i+1] for i in range(len(changed))])
            self.assertEqual(result["completion_tokens"], 2)

    async def test_multi_data_lines_and_comments(self):
        wire = b'event: message\n: keepalive\ndata: {"choices":\ndata: [{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        result = await self.parse(wire + event(usage(0)) + b"data: [DONE]\n\n")
        self.assertEqual(result["completion_tokens"], 0)
        self.assertIsNone(result["ttft"])

    async def test_invalid_token_types(self):
        for tokens in [True, -1, 1.0, "1", None, 0]:
            with self.subTest(tokens=tokens):
                await self.reject(event(choice("Text", "stop")) + event(usage(tokens)) + b"data: [DONE]\n\n", "invalid_completion_tokens")

    async def test_final_usage_must_follow_finish_once(self):
        await self.reject(event(usage(2)), "invalid_final_usage")
        wire = event(choice("Text", "stop")) + event(usage(2))
        await self.reject(wire + event(usage(2)), "invalid_final_usage")

    async def test_nonterminal_usage_not_trusted(self):
        obj = choice("Text")
        obj["usage"] = {"completion_tokens": 2}
        await self.reject(event(obj), "nonterminal_usage")

    async def test_no_data_or_choice_after_terminal_state(self):
        await self.reject(fixture("normal.sse") + event(choice("extra")), "data_after_done")
        await self.reject(event(choice("Text", "stop")) + event(choice("extra")), "choice_after_finish")

    async def test_duplicate_json_key_rejected(self):
        await self.reject(b'data: {"choices":[],"choices":[]}\n\n', "duplicate_json_key")

    async def test_invalid_or_multiple_choices(self):
        for obj in [{}, {"choices": {}}, {"choices": [choice()["choices"][0]] * 2}]:
            with self.subTest(obj=obj):
                await self.reject(event(obj), "invalid_choices")
        obj = choice("Text", "stop")
        obj["choices"][0]["index"] = 1
        await self.reject(event(obj), "invalid_choice")

    async def test_error_finish_and_nontext_delta_rejected(self):
        await self.reject(event(choice("Text", "error")), "unsupported_finish_reason")
        obj = choice()
        obj["choices"][0]["delta"]["tool_calls"] = [{"index": 0}]
        await self.reject(event(obj), "unsupported_delta")
        obj["choices"][0]["delta"] = {"reasoning": "nontext mode"}
        await self.reject(event(obj), "unsupported_delta")

    async def test_bounded_frame(self):
        await self.reject(b"data: " + b"x" * (1024 * 1024 + 1), "frame_too_large")


class RequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_500_rejected_before_body_is_read(self):
        response = Response(fixture("http_500.json"), status=500, media="application/json")
        result = await measure_request(Session([response]), "unused", {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "http_error")
        self.assertEqual(result["http_status"], 500)
        self.assertFalse(response.content.iterated)

    async def test_wrong_media_type_rejected(self):
        result = await measure_request(Session([Response(fixture("normal.sse"), media="application/json")]), "unused", {})
        self.assertEqual(result["error"], "invalid_content_type")

    async def test_disconnection_and_timeout_are_failures_not_partial_success(self):
        for wire in [fixture("no_done.sse"), fixture("normal.sse")]:
            for error in [OSError("private path"), TimeoutError("secret"), RuntimeError("payload aborted")]:
                with self.subTest(error=type(error).__name__, done=b"[DONE]" in wire):
                    result = await measure_request(Session([Response(wire, error=error)]), "unused", {})
                    self.assertEqual(result["error"], "transport_error")
                    self.assertNotIn("private", str(result))
                    self.assertNotIn("secret", str(result))

    async def test_both_entrypoints_request_final_usage_n1(self):
        for function, args in [(bench_serving.one_request, ("prompt", 8, False)), (quick_bench.one, (1, 8))]:
            with self.subTest(entrypoint=function.__module__):
                session = Session([Response(fixture("normal.sse"))])
                result = await function(session, *args)
                self.assertEqual(result["completion_tokens"], 7)
                self.assertEqual(session.requests[0]["stream_options"], {"include_usage": True})
                self.assertEqual(session.requests[0]["n"], 1)

    async def test_both_entrypoints_propagate_protocol_failure_and_unavailable(self):
        for function, args in [(bench_serving.one_request, ("prompt", 8, False)), (quick_bench.one, (1, 8))]:
            for name, expected in [("no_done.sse", "missing_done"), ("server_error.sse", "server_error"), ("missing_usage.sse", None)]:
                with self.subTest(entrypoint=function.__module__, fixture=name):
                    result = await function(Session([Response(fixture(name))]), *args)
                    self.assertEqual(result["error"], expected)
                    self.assertEqual(result["ok"], expected is None)
                    if expected is None:
                        self.assertIsNone(result["completion_tokens"])

    async def test_both_batches_reject_failed_warmup(self):
        for module, function, args in [(bench_serving, bench_serving.run_batch, (1, 2, 8, False)), (quick_bench, quick_bench.run, (1, 2, 8))]:
            with self.subTest(module=module.__name__):
                session = Session([Response(status=500)])
                with patch.object(module.aiohttp, "ClientSession", return_value=session):
                    with self.assertRaisesRegex(RuntimeError, "warmup_failed: http_error"):
                        await function(*args)
                self.assertEqual(len(session.requests), 1)

    async def test_both_batches_preserve_mixed_failure_counts(self):
        for module, function, args in [(bench_serving, bench_serving.run_batch, (1, 2, 8, False)), (quick_bench, quick_bench.run, (1, 2, 8))]:
            with self.subTest(module=module.__name__):
                session = Session([Response(fixture("normal.sse")), Response(fixture("normal.sse")), Response(status=500)])
                with patch.object(module.aiohttp, "ClientSession", return_value=session):
                    result = await function(*args)
                self.assertEqual((result["attempted"], result["successful"], result["failed"]), (2, 1, 1))
                self.assertIsNone(result["tput"])
                self.assertEqual(result["errors"], {"http_error": 1})


class AggregateTests(unittest.TestCase):
    def successful(self, tokens=7, events=2, ttft=1, last=3):
        return {"ok": True, "error": None, "completion_tokens": tokens, "content_events": events, "ttft": ttft, "last_content": last}

    def test_server_token_sum_and_estimate(self):
        result = summarize_requests([self.successful(), self.successful()], 2)
        self.assertEqual(result["tot_tok"], 14)
        self.assertEqual(result["tput"], 7)
        self.assertEqual(result["tpot_estimate_p50"], 2 / 6)

    def test_missing_usage_makes_token_metrics_unavailable(self):
        result = summarize_requests([self.successful(), self.successful(tokens=None)], 2)
        self.assertEqual(result["tokens_unavailable"], 1)
        self.assertIsNone(result["tput"])
        self.assertIsNone(result["tot_tok"])
        self.assertEqual(result["measurement_status"], "incomplete")

    def test_all_failed_no_zero_latency_success(self):
        result = summarize_requests([{"ok": False, "error": "http_error"}], 2)
        self.assertEqual(result["rps_success"], 0)
        for key in ["ttft_p50", "ttft_p99", "tpot_estimate_p50", "tput", "tot_tok"]:
            self.assertIsNone(result[key])

    def test_zero_tokens_and_coalesced_content_do_not_invent_tpot(self):
        result = summarize_requests([self.successful(tokens=0, events=0, ttft=None, last=None)], 2)
        self.assertEqual(result["tput"], 0)
        self.assertIsNone(result["ttft_p50"])
        self.assertIsNone(result["tpot_estimate_p50"])
        self.assertIsNone(summarize_requests([self.successful(events=1)], 2)["tpot_estimate_p50"])

    def test_nearest_rank_p99(self):
        result = summarize_requests([self.successful(ttft=i) for i in range(1, 17)], 2)
        self.assertEqual(result["ttft_p99"], 16)

    def test_invalid_batch_bounds(self):
        for wall in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                summarize_requests([self.successful()], wall)
        with self.assertRaises(ValueError):
            summarize_requests([], 1)


class CLITests(unittest.TestCase):
    def test_incomplete_sweep_exit2_no_gain_and_separate_output(self):
        result = summarize_requests([{"ok": False, "error": "http_error"}], 1)
        async def batch(*args, **kwargs):
            return dict(result, conc=args[0])
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.chdir(directory), patch.object(bench_serving, "run_batch", side_effect=batch):
                Path("sweep_results.json").write_text("retained historical output")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = asyncio.run(bench_serving.sweep(type("Args", (), {"max_tokens": 8})()))
                self.assertEqual(code, 2)
                self.assertNotIn("实测吞吐比", output.getvalue())
                self.assertEqual(Path("sweep_results.json").read_text(), "retained historical output")
                self.assertIsNone(json.loads(Path("sweep_results_v2.json").read_text())[0]["tput"])

    def test_cli_incomplete_and_warmup_failures_exit2(self):
        incomplete = summarize_requests([{"ok": False, "error": "http_error"}], 1)
        for module, funcname in [(quick_bench, "run"), (bench_serving, "sweep")]:
            with self.subTest(module=module.__name__):
                async def failed(*args):
                    raise RuntimeError("warmup_failed: http_error")
                with patch.object(module, funcname, side_effect=failed), patch.object(sys, "argv", [module.__name__]), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(module.main(), 2)
        async def run(*args):
            return incomplete
        with patch.object(quick_bench, "run", side_effect=run), patch.object(sys, "argv", ["quick_bench"]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(quick_bench.main(), 2)

    def test_nonpositive_arguments_exit_before_session(self):
        for script, arg in [("bench_serving.py", "--max-tokens=0"), ("quick_bench.py", "0")]:
            result = subprocess.run([sys.executable, "-B", str(ROOT / script), arg], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("must be a positive integer", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_successful_cli_keeps_undefined_zero_baseline_ratio_unavailable(self):
        empty = summarize_requests([{"ok": True, "error": None, "completion_tokens": 0,
                                    "content_events": 0, "ttft": None, "last_content": None}], 1)
        async def batch(*args, **kwargs):
            return dict(empty, conc=args[0])
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            output = io.StringIO()
            with patch.object(bench_serving, "run_batch", side_effect=batch), contextlib.redirect_stdout(output):
                self.assertEqual(asyncio.run(bench_serving.sweep(type("Args", (), {"max_tokens": 8})())), 0)
            self.assertIn("基线输出 token 为零", output.getvalue())
            self.assertNotIn("实测吞吐比", output.getvalue())


if __name__ == "__main__":
    unittest.main()
