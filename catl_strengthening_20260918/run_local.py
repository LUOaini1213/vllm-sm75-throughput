"""Offline two-precision experiment using the existing WSL vLLM environment.

Launches only loopback; owns and cleans up only its own server process group.
No new model download, cloud GPU, or production service is used.
"""
import asyncio
import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from benchmark import run_sweep, run_reliability
from quality_eval import evaluate, compare

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8018
BASE = f"http://127.0.0.1:{PORT}"


def command_output(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "transformers-reference"], default="vllm")
    args = parser.parse_args()
    import torch
    import transformers
    import vllm
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = HERE / "evidence" / ("local_" + stamp)
    dest.mkdir(parents=True, exist_ok=False)
    # Fail fast rather than touching an unrelated process using our chosen port.
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError("Port already occupied; refusing to reuse existing server")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "VLLM_WSL2_ENABLE_PIN_MEMORY": "1", "VLLM_USE_V2_MODEL_RUNNER": "0",
           "PYTHONIOENCODING": "utf-8"}
    manifest = {"started_utc": stamp, "platform": platform.platform(), "python": sys.version,
                "torch": torch.__version__, "vllm": vllm.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                "nvidia_smi_before": command_output(["nvidia-smi"]), "model": MODEL,
                "offline": True, "backend": args.backend, "vllm_use_v2_model_runner": False, "experiments": [],
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
                "quality_dataset_sha256": hashlib.sha256((HERE / "quality_cases.json").read_bytes()).hexdigest()}
    qualities = []
    try:
        for dtype in ["half", "float"]:
            label = "fp16" if dtype == "half" else "fp32"
            command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
                       "--dtype", dtype, "--max-model-len", "512", "--max-num-seqs", "4",
                       "--gpu-memory-utilization", "0.75", "--enforce-eager", "--attention-backend", "TRITON_ATTN",
                       "--host", "127.0.0.1", "--port", str(PORT)]
            if args.backend == "transformers-reference":
                command = [sys.executable, str(HERE / "reference_service.py"), "--dtype", dtype,
                           "--device", "cuda", "--port", str(PORT)]
            experiment = {"label": label, "command": command}
            manifest["experiments"].append(experiment)
            with (dest / f"{label}_server.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                started = time.monotonic()
                try:
                    while time.monotonic() - started < 360:
                        if process.poll() is not None:
                            raise RuntimeError(f"{label} server exited {process.returncode}")
                        try:
                            with urllib.request.urlopen(BASE + "/v1/models", timeout=2) as response:
                                experiment["models_response"] = json.load(response)
                            break
                        except (OSError, urllib.error.URLError):
                            time.sleep(1)
                    else:
                        raise TimeoutError(f"{label} startup exceeded 360 seconds")
                    experiment["ready_s"] = time.monotonic() - started
                    print(f"READY {label} after {experiment['ready_s']:.1f}s -> {dest}", flush=True)
                    sweep = asyncio.run(run_sweep(BASE + "/v1/chat/completions", MODEL, dest / f"{label}_sweep.json"))
                    experiment["sweep_successes"] = sum(b["summary"]["statuses"].get("success", 0) for b in sweep["batches"])
                    reliability = asyncio.run(run_reliability(BASE + "/v1/chat/completions", MODEL, dest / f"{label}_reliability.json"))
                    experiment["reliability"] = {"passed": reliability["passed"], "total": reliability["total"]}
                    quality_path = dest / f"{label}_quality.json"
                    quality = asyncio.run(evaluate(BASE + "/v1/chat/completions", MODEL, label, quality_path))
                    qualities.append(quality_path)
                    experiment["quality"] = {"exact_matches": quality["exact_matches"], "total": quality["total"]}
                    experiment["nvidia_smi_after"] = command_output(["nvidia-smi"])
                    print(f"COMPLETE {label}: {experiment['reliability']} {experiment['quality']}", flush=True)
                except Exception as exc:
                    experiment["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                    experiment["shutdown_returncode"] = process.returncode
                    time.sleep(2)
        (dest / "precision_quality_comparison.json").write_text(json.dumps(compare(qualities), indent=2), encoding="utf-8")
        manifest["completed"] = True
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["nvidia_smi_final"] = command_output(["nvidia-smi"])
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Evidence:", dest, flush=True)


if __name__ == "__main__":
    main()
