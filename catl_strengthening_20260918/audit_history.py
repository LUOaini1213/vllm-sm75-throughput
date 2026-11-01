"""Recompute historical observations; never label them as a new GPU run."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "evidence"


def main():
    OUT.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, str(REPO / "check_claims.py")],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    (OUT / "historical_check_claims.log").write_bytes(proc.stdout)
    if proc.returncode:
        raise RuntimeError("Existing historical claims check failed")
    spec = importlib.util.spec_from_file_location("historical_claims", REPO / "check_claims.py")
    claims = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(claims)
    rows = claims.parse_notebook()
    findings = []
    for group, cells in rows.items():
        for concurrency, row in cells.items():
            requests = max(4 * concurrency, 16)
            expected = requests / concurrency * (row["ttft"] + 127 * row["tpot"]) / 1000
            findings.append({"group": group, "concurrency": concurrency, **row,
                "tokens_per_request_inferred": row["wall"] * row["tput"] / requests,
                "wall_reconstructed_s": expected,
                "wall_residual_s": row["wall"] - expected,
                "residual_fraction_of_observed": abs(row["wall"] - expected) / row["wall"],
                "flagged": abs(row["wall"] - expected) / row["wall"] > .15})
    a, c = rows["A_default"][32], rows["C_maxbs32"][32]
    expected_a = 4 * (a["ttft"] + 127 * a["tpot"]) / 1000
    inputs = ["check_claims.py", "cloud_dip_confirm_executed.ipynb", "bench_serving.py",
              "results_t4/dip_causal_2026-09-08.txt", "results_t4/corrections_2026-09-13.txt"]
    report = {"kind": "historical_recalculation_no_new_measurement", "rows": findings,
              "preferred_historical_tpot_speedup": a["tpot"] / c["tpot"],
              "raw_throughput_ratio_not_clean_speedup": c["tput"] / a["tput"],
              "estimated_wall_residual_A32_s": a["wall"] - expected_a,
              "old_metric_limitation": "Historical client counts nonempty content SSE events as tokens; no per-request token IDs retained. TPOT is the historical proxy, not revalidated token timing.",
              "sha256_inputs": {str(p): hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in inputs}}
    (OUT / "historical_recalculation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"historical_rows": len(findings), "flagged": [f'{x["group"]}@{x["concurrency"]}' for x in findings if x["flagged"]],
                      "preferred_historical_tpot_speedup": report["preferred_historical_tpot_speedup"]}, indent=2))


if __name__ == "__main__":
    main()
