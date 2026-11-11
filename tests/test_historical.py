import json
from pathlib import Path
import shutil
import tempfile
import unittest

import contextlib
import io
from unittest.mock import patch

import verify_historical
from verify_historical import MANIFEST, verify_history, source_commit_available

ROOT = Path(__file__).resolve().parents[1]


class HistoricalTests(unittest.TestCase):
    SOURCE = "d93344334567f7292099e6b7f1d661fee5f5f7b5"

    @unittest.skipUnless(source_commit_available(ROOT, SOURCE),
                         "source commit not in local history (shallow clone); CI runs this with full history")
    def test_original_bytes_and_historical_files(self):
        result = verify_history(ROOT, git_source=True)
        self.assertEqual(result["frozen_scripts"], 3)
        self.assertEqual(result["unchanged_historical_files"], 29)
        self.assertEqual(result["source_commit"], self.SOURCE)

    def test_hash_layer_counts_without_git(self):
        result = verify_history(ROOT)
        self.assertEqual((result["frozen_scripts"], result["unchanged_historical_files"]), (3, 29))
        self.assertFalse(result["git_source_checked"])

    def test_isolate_snapshot_is_the_old_chunk_counting_client(self):
        frozen = (ROOT / "historical/d933443/isolate.py.txt").read_text(encoding="utf-8")
        self.assertIn("n+=1", frozen)
        self.assertNotIn("serving_metrics", frozen)
        self.assertIn("measure_request", (ROOT / "isolate.py").read_text(encoding="utf-8"))

    def test_shallow_clone_skips_git_layer_unless_required(self):
        err = io.StringIO()
        with patch.object(verify_historical, "source_commit_available", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(err):
            self.assertEqual(verify_historical.main([]), 0)
            self.assertIn("SKIPPED git-source comparison", err.getvalue())
            self.assertIn('"git_source_checked": false', out.getvalue())
            self.assertEqual(verify_historical.main(["--require-git-source"]), 2)
            self.assertIn("not in local history", err.getvalue())

    def make_copy(self, target):
        manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
        for path in [MANIFEST] + [entry["path"] for entry in manifest["snapshots"] + manifest["unchanged_files"]]:
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, destination)

    def test_missing_or_changed_frozen_script_and_raw_fail_closed(self):
        for path in ["historical/d933443/bench_serving.py.txt", "results_t4/t4_serving_and_prefix.txt"]:
            for change in ["missing", "modified"]:
                with self.subTest(path=path, change=change), tempfile.TemporaryDirectory() as directory:
                    target = Path(directory)
                    self.make_copy(target)
                    file = target / path
                    if change == "missing":
                        file.unlink()
                    else:
                        file.write_bytes(file.read_bytes() + b"tampered")
                    with self.assertRaisesRegex(ValueError, "historical artifact missing or changed"):
                        verify_history(target)

    def test_manifest_cannot_resign_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            self.make_copy(target)
            path = target / MANIFEST
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                verify_history(target)


if __name__ == "__main__":
    unittest.main()
