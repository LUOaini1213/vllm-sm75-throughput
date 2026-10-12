import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_historical import MANIFEST, verify_history

ROOT = Path(__file__).resolve().parents[1]


class HistoricalTests(unittest.TestCase):
    def test_original_bytes_and_historical_files(self):
        result = verify_history(ROOT, git_source=True)
        self.assertEqual(result["frozen_scripts"], 2)
        self.assertEqual(result["unchanged_historical_files"], 30)
        self.assertEqual(result["source_commit"], "d93344334567f7292099e6b7f1d661fee5f5f7b5")

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
