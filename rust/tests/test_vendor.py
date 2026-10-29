"""Vendoring guard tests; use temporary Git repositories, not real submodules."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "vendor", Path(__file__).resolve().parents[1] / "scripts/vendor.py")
assert spec is not None and spec.loader is not None
vendor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vendor)


class VendorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="clap-vendor-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.crate = self.root / "rust"
        self.crate.mkdir()
        self.patch = patch.multiple(vendor, ROOT=self.root, CRATE=self.crate,
                                    SUBMODULES=("ggml",), TREES=("include", "ggml/LICENSE"))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        (self.root / "include").mkdir()
        (self.root / "include/clap.h").write_text("header\n")
        (self.root / "python/tests").mkdir(parents=True)
        (self.root / "python/tests/tiny_models.py").write_text("# fixture\n")
        self.git("init", "-q")
        self.module = self.root / "ggml"
        self.module.mkdir()
        self.git("init", "-q", cwd=self.module)
        (self.module / "LICENSE").write_text("license\n")
        self.commit(self.module)
        self.git("add", "ggml")
        self.commit(self.root)

    def git(self, *args, cwd=None):
        return subprocess.check_output(["git", *args], cwd=cwd or self.root,
                                       stderr=subprocess.PIPE, text=True).strip()

    def commit(self, cwd):
        self.git("add", "-A", cwd=cwd)
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "fixture", cwd=cwd)

    def test_repeatable_copy_and_license(self):
        vendor.main()
        first = {p.relative_to(self.crate / "vendor").as_posix(): p.read_bytes()
                 for p in (self.crate / "vendor").rglob("*") if p.is_file()}
        (self.crate / "vendor/stale").write_text("remove me")
        vendor.main()
        second = {p.relative_to(self.crate / "vendor").as_posix(): p.read_bytes()
                  for p in (self.crate / "vendor").rglob("*") if p.is_file()}
        self.assertEqual(first, second)
        self.assertEqual(first["ggml/LICENSE"], (self.module / "LICENSE").read_bytes())
        self.assertIn("test-data/tiny_models.py", first)
        self.assertIn("revisions.json", first)

    def test_rejects_dirty_module_without_replacing_vendor(self):
        (self.crate / "vendor").mkdir()
        sentinel = self.crate / "vendor/sentinel"
        sentinel.write_text("keep")
        (self.module / "LICENSE").write_text("dirty\n")
        with self.assertRaisesRegex(SystemExit, "dirty submodule"):
            vendor.main()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_rejects_wrong_revision(self):
        (self.module / "LICENSE").write_text("new revision\n")
        self.commit(self.module)
        with self.assertRaisesRegex(SystemExit, "expected pinned"):
            vendor.main()

    def test_rejects_uninitialized_module(self):
        (self.module / ".git").rename(self.module / "hidden-git")
        with self.assertRaisesRegex(SystemExit, "submodule update"):
            vendor.main()


if __name__ == "__main__":
    unittest.main()
