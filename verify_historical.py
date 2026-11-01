"""Verify frozen client identity and unchanged historical artifacts, no replay."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

MANIFEST = "historical/d933443/manifest.json"
MANIFEST_SHA256 = "be533ac9e83ed6c1a89c913b7679cff7d266315df2bdacc3f79b628ea51a27ab"


def verify_history(root, *, git_source=False):
    root = Path(root)
    raw = (root / MANIFEST).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("historical manifest hash mismatch")
    manifest = json.loads(raw)
    entries = manifest["snapshots"] + manifest["unchanged_files"]
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("historical artifact missing or changed: " + entry["path"])
    if git_source:
        for entry in manifest["snapshots"]:
            original = subprocess.check_output(
                ["git", "show", manifest["source_commit"] + ":" + entry["original_path"]], cwd=root)
            if original != (root / entry["path"]).read_bytes():
                raise ValueError("snapshot differs from recorded Git source: " + entry["path"])
    return {"source_commit": manifest["source_commit"], "frozen_scripts": len(manifest["snapshots"]),
            "unchanged_historical_files": len(manifest["unchanged_files"]),
            "manifest_sha256": MANIFEST_SHA256, "git_source_checked": git_source}


def main():
    try:
        result = verify_history(Path(__file__).resolve().parent, git_source=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(dict(result, status="PASS"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
