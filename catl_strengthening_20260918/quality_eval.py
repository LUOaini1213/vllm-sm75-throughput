"""Fixed local engineering smoke tasks, not a general model capability benchmark.

Can compare named fp16/AWQ/GPTQ endpoints without downloading or serving them.
An unavailable variant stays unavailable; text similarity is not quality.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import aiohttp
from benchmark import body, request

HERE = Path(__file__).resolve().parent


def normalize(text):
    return text.strip().rstrip(".").strip()


async def evaluate(url, model, label, output):
    data = (HERE / "quality_cases.json").read_bytes()
    cases = json.loads(data)
    results = []
    async with aiohttp.ClientSession() as session:
        for case in cases:
            record = await request(session, url, body(model, case["prompt"], 48), request_id=case["id"])
            results.append({**case, "record": record,
                            "exact_match": record["status"] == "success" and normalize(record["output_text"]) == case["answer"]})
    out = {"label": label, "model": model, "url": url,
           "dataset_sha256": hashlib.sha256(data).hexdigest(),
           "kind": "fixed_local_smoke_tasks_not_general_accuracy", "results": results,
           "successes": sum(r["record"]["status"] == "success" for r in results),
           "exact_matches": sum(r["exact_match"] for r in results), "total": len(results)}
    Path(output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def compare(paths):
    variants = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if len({v["dataset_sha256"] for v in variants}) != 1:
        raise ValueError("Different prompt sets cannot be compared")
    if len({v["total"] for v in variants}) != 1:
        raise ValueError("Incomplete prompt sets cannot be compared")
    return {"kind": "paired_fixed_task_comparison", "dataset_sha256": variants[0]["dataset_sha256"],
            "variants": [{"label": v["label"], "model": v["model"], "successes": v["successes"],
                          "exact_matches": v["exact_matches"], "total": v["total"]} for v in variants],
            "paired": [{"id": x["id"], "matches": {v["label"]: next(r["exact_match"] for r in v["results"] if r["id"] == x["id"]) for v in variants}} for x in variants[0]["results"]]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8018/v1/chat/completions")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--label", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    asyncio.run(evaluate(args.url, args.model, args.label, args.output))
