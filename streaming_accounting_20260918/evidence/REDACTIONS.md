# Edits to the raw evidence (2026-11-03)

The run logs and manifests in this folder are kept as they were written, with one
exception. Three files contained the absolute path of the author's local checkout
(a WSL `/mnt/c/Users/...` path, including the old directory name). On 2026-11-03
that prefix was replaced with `<repo>/streaming_accounting_20260918`, and the
directory itself was renamed from its earlier name to `streaming_accounting_20260918/`:

| File | Occurrences replaced |
|---|---:|
| `local_20260918T054332Z/fp16_server.log` | 1 line (traceback `File "..."`) |
| `local_20260918T054332Z/manifest.json` | 1 (`experiments[0].command[1]`) |
| `local_20260918T054433Z/manifest.json` | 2 (`experiments[*].command[1]`) |

No measured value, timestamp, token count or status changed. Because
`local_20260918T054433Z/manifest.json` changed, its sha256 in `summary.json`
changed too; `summary.json` and `../RESULTS.md` were regenerated with
`python analyze_results.py`, and `python analyze_results.py --check` (run in CI)
confirms both still follow from the raw files. The earlier bytes remain in the
Git history of this repository.
