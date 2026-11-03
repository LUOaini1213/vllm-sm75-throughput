import unittest
from benchmark import SSEParser, finalize, percentile, summarize


class BenchmarkTests(unittest.TestCase):
    def test_sse_utf8_fragmentation_and_multiline(self):
        raw = ': ping\r\ndata: {"text": "交通",\r\ndata: "id": 1}\r\n\r\ndata: [DONE]\n\n'.encode()
        parser = SSEParser()
        actual = []
        for byte in raw:
            actual.extend(parser.feed(bytes([byte])))
        self.assertEqual(actual, ['{"text": "交通",\n"id": 1}', '[DONE]'])

    def record(self, events, *, usage=3, done=True):
        return {"events": events, "usage": {"completion_tokens": usage}, "done": done,
                "finish_reason": "length", "status": "pending", "client_queue_ms": 500}

    def test_token_ids_not_content_chunks_and_queue(self):
        r = finalize(self.record([
            {"elapsed_s": .1, "content": "", "token_ids": [1]},
            {"elapsed_s": .12, "content": "hi", "token_ids": [2]},
            {"elapsed_s": .15, "content": "!", "token_ids": [3]}]))
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["content_event_count"], 2)
        self.assertEqual(r["stream_token_count"], 3)
        self.assertEqual(r["ttft_including_client_queue_ms"], 600)
        self.assertAlmostEqual(r["tpot_ms"], 25)

    def test_batched_token_latency_not_fabricated(self):
        r = finalize(self.record([{"elapsed_s": .1, "content": "Hi", "token_ids": [1, 2]},
                                 {"elapsed_s": .15, "content": "!", "token_ids": [3]}]))
        self.assertIsNone(r["tpot_ms"])
        self.assertEqual(r["inter_token_arrival_ms"], [])

    def test_missing_usage_is_not_success(self):
        r = finalize(self.record([{"elapsed_s": .1, "content": "x", "token_ids": [1]}], usage=None))
        self.assertEqual(r["status"], "token_accounting_error")

    def test_truncated_stream_is_not_success(self):
        r = finalize(self.record([{"elapsed_s": .1, "content": "x", "token_ids": [1]}], usage=1, done=False))
        self.assertEqual(r["status"], "incomplete_stream")

    def test_failure_excluded_from_goodput(self):
        r = finalize(self.record([{"elapsed_s": .1, "content": "x", "token_ids": [1]}], usage=1))
        failed = {**r, "status": "client_timeout", "stream_token_count": 99}
        summary = summarize([r, failed], 2)
        self.assertEqual(summary["successful_tokens_per_s"], .5)
        self.assertEqual(summary["success_rate"], .5)

    def test_percentile_edge_cases(self):
        self.assertIsNone(percentile([], .99))
        self.assertEqual(percentile([7], .99), 7)
        self.assertEqual(percentile(list(range(1, 101)), .99), 99)


if __name__ == "__main__":
    unittest.main()
