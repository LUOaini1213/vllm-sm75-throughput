"""Hash already-cached weights. Never downloads models."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
snapshots = list((root / "snapshots").iterdir())
out = []
for snapshot in snapshots:
    files = {}
    for name in ["config.json", "generation_config.json", "tokenizer.json", "model.safetensors"]:
        path = snapshot / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    out.append({"model": "Qwen/Qwen2.5-0.5B-Instruct", "snapshot_revision": snapshot.name,
                "configuration": json.loads((snapshot / "config.json").read_text()),
                "generation_configuration": json.loads((snapshot / "generation_config.json").read_text()),
                "files": files, "new_download": False})
(HERE / "evidence/cached_model_inventory.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps([{"revision": r["snapshot_revision"], "weights_bytes": r["files"]["model.safetensors"]["bytes"]} for r in out]))
