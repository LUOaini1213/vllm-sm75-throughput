# Streaming client integrity (2026-10-12)

The revised clients validate protocol completion before accepting a request.
This is an offline reliability change, with no model, GPU, real-server benchmark,
or new performance measurement. Synthetic wire fixtures exercise the real parser
through a fake transport; they do not prove compatibility with a running server
or that historical measurements were affected.

Both `bench_serving.py` and `quick_bench.py` request `n=1`, streaming, and
`stream_options.include_usage=true`. The final-only usage format follows the
[vLLM v0.28.0 chat streaming implementation](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/entrypoints/openai/chat_completion/serving.py):
the terminal usage chunk has `choices=[]` and is followed by `[DONE]`.
Earlier [v0.6.6 source](https://github.com/vllm-project/vllm/blob/v0.6.6/vllm/entrypoints/openai/serving_chat.py)
uses the same final-only format. Neither source is a runtime compatibility test.

The supported mode is single-choice text chat. A request succeeds only with a
2xx HTTP status, `text/event-stream` content type, valid UTF-8/JSON SSE frames,
a supported `stop`/`length` finish, `[DONE]`, and clean EOF. HTTP failures,
server error frames (even followed by `[DONE]`), malformed or truncated frames,
missing finish/DONE, and transport exceptions are explicit failures. Errors are
safe diagnostic codes; response text and arbitrary exception details are not
printed. Comments, multiline data, LF/CRLF/CR, and split UTF-8 are supported.
Frames above 1 MiB are rejected. Tools, continuous usage, multiple choices, and
nontext deltas are outside this benchmark mode and rejected.

Only a terminal server `usage.completion_tokens` integer supplies token totals.
This trusts the server's reported count; it is not an independent tokenizer or
server implementation proof. Content events are counted separately and never
treated as tokens. Missing final usage permits protocol completion but records
`completion_tokens=null`, `token_source=unavailable`. Invalid, duplicate,
nonterminal, negative, boolean, or inconsistent zero-with-content counts fail.

Batch schema v2 reports attempted/successful/failed requests, error-code counts,
successful/attempted RPS, and missing usage counts. RPS uses the whole measured
batch wall time, including failed attempts; warmup is excluded. Any failed
request or missing token count makes overall `tot_tok`/`tput` unavailable. The
clients return exit 2 for incomplete batches or failed warmup and do not print
a throughput gain. Latency summaries use successful requests with observations
and expose their observation counts, so an incomplete batch is not a zero-latency
success. TTFT p99 uses nearest rank. Empty-output completions may have zero tokens
and no latency observation.

TTFT means time to the first nonempty content event observed by this client.
`tpot_estimate_p50` uses `(last_content_time - first_content_time) /
(completion_tokens - 1)` only when at least two content events and two tokens
exist. This is an event-timed estimate: coalesced events and network buffering
prevent recovery of exact per-token decode latency. Its observation count may
be smaller than the successful request count; a single coalesced event gives
no estimate. These labels deliberately differ from historical `tpot_p50`.

For an already-running authorized server, the revised entrypoints are:

```bash
python bench_serving.py
python bench_serving.py --prefix-test
python quick_bench.py 8
```

The sweep writes `sweep_results_v2.json`; prefix input comparison writes
`prefix_results_v2.json`. `quick_bench.py` prints schema v2 JSON. These do not
overwrite the old `sweep_results.json` name. Rows include the entrypoint/shared
parser SHA256 and request settings. Prefix mode changes the input, so a causal
prefix-cache experiment still requires matched input and independently controlled
server caching on/off. Existing historical controlled experiments remain intact.

Offline checks (Python 3.11+, `aiohttp==3.14.3`; no network requests):

```bash
python -B -m unittest discover -s tests -v
python -O -B -m unittest discover -s tests -v
python -B verify_historical.py
python -O -B verify_historical.py
```

The historical manifest is bound by a SHA256 constant in `verify_historical.py`.
It protects 30 existing historical files (all raw `results_t4`, notebooks, other
source/docs/license) and two frozen script snapshots. The checker also compares
the snapshots byte-for-byte with recorded Git source
`d93344334567f7292099e6b7f1d661fee5f5f7b5`; that commit must be available locally
(CI checks out history). Missing files, changed bytes, or changed manifests fail
closed, including under optimized Python. No old metric is recomputed.
`.gitattributes` preserves recorded bytes instead of converting line endings on
Windows; an existing checkout with converted line endings must restore the
verified original files before a byte audit can pass.

The `.py.txt` snapshots are audit artifacts only, are not imported or executed,
and cannot validate the revised runtime or support new performance claims.
They freeze the last repository source, not a newly authenticated receipt for
every old session. The frozen benchmark SHA256 begins `ef5bf10717ecdacf`, matching
the README's recorded short hash; that identity does not authenticate every old
response or establish absence/presence of protocol failures. Actual offline
test counts and current source hashes are recorded in
[serving-integrity-checks.json](serving-integrity-checks.json).
