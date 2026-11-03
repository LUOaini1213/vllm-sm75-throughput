"""Verify frozen client identity and unchanged historical artifacts, no replay.

Two layers:
  1. sha256 of every manifest entry (works in any checkout, including a ZIP).
  2. each frozen snapshot equals `git show <source_commit>:<original_path>`.
Layer 2 needs the source commit in local history. In a shallow clone
(`git clone --depth 1`) or a ZIP download that commit is absent: the script then
reports layer 2 as SKIPPED, says how to get the history, and still exits 0 if
layer 1 passes. `--require-git-source` (used in CI) turns that skip into a failure.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

MANIFEST = "historical/d933443/manifest.json"
MANIFEST_SHA256 = "f4bdd1584d92f457fc5871da173cf19aea67917c26fc55d434afcca95bcbd052"


def source_commit_available(root, commit):
    """True when `commit` is present in the local Git object store."""
    try:
        return subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=root,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parent
    require = "--require-git-source" in argv
    try:
        commit = json.loads((root / MANIFEST).read_bytes())["source_commit"]
        have_git = source_commit_available(root, commit)
        if require and not have_git:
            raise ValueError("source commit %s not in local history (clone without --depth)" % commit)
        result = verify_history(root, git_source=have_git)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        return 2
    if not have_git:
        print("SKIPPED git-source comparison: commit %s is not in local history "
              "(shallow clone or ZIP). Hashes were still checked. For the full check: "
              "git fetch --unshallow" % commit[:7], file=sys.stderr)
    print(json.dumps(dict(result, status="PASS"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
