"""isolate.py on the shared client: offline wire-byte fixtures, no real server."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import isolate
from serving_metrics import summarize_requests
from test_serving_metrics import Response, Session, fixture


class IsolateRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_final_usage_and_counts_server_tokens(self):
        session = Session([Response(fixture("normal.sse"))])
        result = await isolate.one(session, 1, 64)
        # normal.sse has 2 non-empty content events but the server reports 7 tokens:
        # the historical script would have counted 2.
        self.assertEqual(result["completion_tokens"], 7)
        self.assertEqual(result["content_events"], 2)
        self.assertEqual(session.requests[0]["stream_options"], {"include_usage": True})
        self.assertEqual(session.requests[0]["n"], 1)
        self.assertEqual(session.requests[0]["max_tokens"], 64)

    async def test_protocol_failures_are_failures(self):
        for name, expected in [("no_done.sse", "missing_done"), ("server_error.sse", "server_error"),
                               ("missing_usage.sse", None)]:
            with self.subTest(fixture=name):
                result = await isolate.one(Session([Response(fixture(name))]), 1, 64)
                self.assertEqual(result["error"], expected)
                self.assertEqual(result["ok"], expected is None)
                if expected is None:
                    self.assertIsNone(result["completion_tokens"])

    async def test_http_status_is_checked(self):
        result = await isolate.one(Session([Response(status=500)]), 1, 64)
        self.assertEqual((result["ok"], result["error"], result["http_status"]), (False, "http_error", 500))

    async def test_burst_rejects_failed_warmup(self):
        session = Session([Response(status=500)])
        with patch.object(isolate.aiohttp, "ClientSession", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "warmup_failed: http_error"):
                await isolate.burst(2)
        self.assertEqual(len(session.requests), 1)

    async def test_burst_with_one_truncated_stream_has_no_throughput(self):
        session = Session([Response(fixture("normal.sse")), Response(fixture("normal.sse")),
                           Response(fixture("no_done.sse"))])
        with patch.object(isolate.aiohttp, "ClientSession", return_value=session):
            result = await isolate.burst(2)
        self.assertEqual((result["attempted"], result["successful"], result["failed"]), (2, 1, 1))
        self.assertEqual(result["measurement_status"], "incomplete")
        self.assertIsNone(result["tput"])
        self.assertEqual(result["errors"], {"missing_done": 1})

    async def test_complete_burst_uses_usage_tokens(self):
        session = Session([Response(fixture("normal.sse")) for _ in range(4)])
        with patch.object(isolate.aiohttp, "ClientSession", return_value=session):
            result = await isolate.burst(3)
        self.assertEqual(result["measurement_status"], "complete")
        self.assertEqual(result["tot_tok"], 21)
        self.assertEqual(result["burst"], 3)


class IsolateCLITests(unittest.TestCase):
    def test_incomplete_sweep_exits_2_and_writes_separate_file(self):
        incomplete = summarize_requests([{"ok": False, "error": "http_error"}], 1)

        async def burst(n, n_tok=64):
            return dict(incomplete, burst=n)
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            with patch.object(isolate, "burst", side_effect=burst), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(isolate.main(), 2)
            rows = json.loads(Path("isolate_results_v2.json").read_text(encoding="utf-8"))
            self.assertEqual([r["burst"] for r in rows], isolate.BURSTS)
            self.assertTrue(all(r["tput"] is None for r in rows))

    def test_warmup_failure_exits_2(self):
        async def failed(*args, **kwargs):
            raise RuntimeError("warmup_failed: http_error")
        with patch.object(isolate, "burst", side_effect=failed), contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(isolate.main(), 2)

    def test_nonpositive_burst_rejected_before_session(self):
        with self.assertRaises(ValueError):
            asyncio.run(isolate.burst(0))


if __name__ == "__main__":
    unittest.main()
