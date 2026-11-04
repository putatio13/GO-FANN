"""Frozen input verification detects content corruption and missing prerequisites."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.tools.verify_paper_inputs import DEFAULT_MANIFEST, manifest_entries, verify_inputs


class VerifyPaperInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"query-order-and-ground-truth\n" * 100000
        self.entry = {
            "path": "data/query.bin",
            "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        self.manifest = {"schema_version": 1, "packages": [{"name": "fixture", "files": [self.entry]}]}

    def write_payload(self, payload=None):
        path = self.root / self.entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload if payload is None else payload)
        return path

    def test_complete_identity(self):
        self.write_payload()
        report = verify_inputs(self.root, self.manifest)
        self.assertTrue(report["ok"])
        self.assertEqual(report["verified_files"], 1)

    def test_same_size_corruption_fails(self):
        self.write_payload(payload=b"X" + self.payload[1:])
        report = verify_inputs(self.root, self.manifest)
        self.assertFalse(report["ok"])
        self.assertEqual(report["artifacts"][0]["status"], "mismatch")

    def test_truncated_file_fails(self):
        self.write_payload(payload=self.payload[:-1])
        self.assertFalse(verify_inputs(self.root, self.manifest)["ok"])

    def test_missing_required_fails(self):
        report = verify_inputs(self.root, self.manifest)
        self.assertFalse(report["ok"])
        self.assertEqual(report["artifacts"][0]["status"], "missing")

    def test_flat_manifest_rejected(self):
        manifest = {"schema_version": 1, "artifacts": [self.entry]}
        for archives in (False, True):
            with self.subTest(archives=archives), self.assertRaisesRegex(ValueError, "packages"):
                verify_inputs(self.root, manifest, archives=archives)

    def test_empty_inventory_rejected(self):
        self.manifest["packages"][0]["files"] = []
        with self.assertRaises(ValueError):
            verify_inputs(self.root, self.manifest)

    def test_unsafe_path_rejected(self):
        for path in ("../outside", "/outside"):
            self.entry["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                verify_inputs(self.root, self.manifest)

    def test_invalid_file_metadata_rejected(self):
        for field, value in (("path", ""), ("path", None),
                             ("bytes", -1), ("bytes", True), ("bytes", "1"),
                             ("sha256", ""), ("sha256", "g" * 64), ("sha256", None)):
            previous = self.entry[field]
            self.entry[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                verify_inputs(self.root, self.manifest)
            self.entry[field] = previous

    def test_cli_reports_failure_and_success(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.manifest))
        command = [sys.executable, "-m", "tests.tools.verify_paper_inputs",
                   "--root", str(self.root), "--manifest", str(manifest)]
        for expected in (1, 0):
            run = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, expected, run.stderr)
            self.assertEqual(json.loads(run.stdout)["ok"], expected == 0)
            self.write_payload()

    def package_manifest(self):
        def entry(path):
            return {"path": path, "bytes": len(self.payload),
                    "sha256": self.entry["sha256"]}
        return {
            "schema_version": 1,
            "packages": [
                {"name": "queries", "files": [entry("data/laion/query.bin")],
                 "assets": [dict(entry("unused"), name="queries.tar.gz")]},
                {"name": "arxiv", "files": [entry("data/arxiv/labels.bin")],
                 "assets": [dict(entry("unused"), name="metadata.tar.gz.part000"),
                            dict(entry("unused"), name="metadata.tar.gz.part001")]},
            ],
            "checksum_asset": dict(entry("unused"), name="SHA256SUMS"),
        }

    def write_inventory(self, manifest, archives=False):
        for entry in manifest_entries(manifest, archives):
            path = self.root / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.payload)

    def test_packages_cover_all_extracted_inputs(self):
        manifest = self.package_manifest()
        self.write_inventory(manifest)
        report = verify_inputs(self.root, manifest)
        self.assertTrue(report["ok"])
        self.assertEqual(report["verified_files"], 2)
        (self.root / "data/arxiv/labels.bin").unlink()
        self.assertFalse(verify_inputs(self.root, manifest)["ok"])
        self.write_inventory(manifest)
        (self.root / "data/laion/query.bin").write_bytes(b"bad")
        self.assertFalse(verify_inputs(self.root, manifest)["ok"])

    def test_archive_parts_and_checksum_sidecar(self):
        manifest = self.package_manifest()
        self.write_inventory(manifest, archives=True)
        report = verify_inputs(self.root, manifest, archives=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["verified_files"], 4)
        for filename in ("metadata.tar.gz.part001", "SHA256SUMS"):
            path = self.root / filename
            path.unlink()
            self.assertFalse(verify_inputs(self.root, manifest, archives=True)["ok"])
            path.write_bytes(self.payload)
        (self.root / "SHA256SUMS").write_bytes(b"wrong")
        self.assertFalse(verify_inputs(self.root, manifest, archives=True)["ok"])

    def test_package_schema_and_unsafe_paths_rejected(self):
        for packages in ([], {}, [None], [{"files": []}], [{"files": [None]}]):
            with self.subTest(packages=packages), self.assertRaises(ValueError):
                verify_inputs(self.root, {"schema_version": 1, "packages": packages})
        for path in ("../outside", "/outside"):
            manifest = self.package_manifest()
            manifest["packages"][0]["files"][0]["path"] = path
            with self.assertRaises(ValueError):
                verify_inputs(self.root, manifest)
            manifest = self.package_manifest()
            manifest["packages"][0]["assets"][0]["name"] = path
            with self.assertRaises(ValueError):
                verify_inputs(self.root, manifest, archives=True)
        manifest = self.package_manifest()
        manifest["packages"][1]["files"] = manifest["packages"][0]["files"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_inputs(self.root, manifest)
        with self.assertRaises(ValueError):
            verify_inputs(self.root, self.manifest, archives=True)
        manifest = self.package_manifest()
        del manifest["checksum_asset"]
        with self.assertRaises(ValueError):
            verify_inputs(self.root, manifest, archives=True)

    def test_symlink_escape_rejected(self):
        (self.root / "data").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            verify_inputs(self.root, self.manifest)

    def test_default_cli_uses_complete_release_inventory(self):
        manifest = json.loads(DEFAULT_MANIFEST.read_text())
        expected = {entry["path"] for package in manifest["packages"] for entry in package["files"]}
        self.assertTrue(any(path.startswith("data/arxiv/") for path in expected))
        for workload in ("categorical", "range", "string_exact", "string_fuzzy", "hybrid"):
            self.assertTrue(any(f"/queries/{workload}/" in path for path in expected))
        for archives in (False, True):
            command = [sys.executable, "-m", "tests.tools.verify_paper_inputs", "--root", str(self.root)]
            if archives:
                command.append("--archives")
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            report = json.loads(run.stdout)
            paths = {entry["path"] for entry in report["artifacts"]}
            if archives:
                self.assertIn("SHA256SUMS", paths)
                self.assertIn("go-fann-arxiv-metadata.tar.gz", paths)
                self.assertIn("go-fann-laion100m-metadata.tar.gz.part001", paths)
            else:
                self.assertEqual(paths, expected)


if __name__ == "__main__":
    unittest.main()
