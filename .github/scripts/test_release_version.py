import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("release_version", Path(__file__).with_name("release-version.py"))
assert spec is not None and spec.loader is not None
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseVersionTest(unittest.TestCase):
    def test_valid_tags(self):
        for tag in ("0.1.0", "v1.2.3", "v1.2.3-rc.1", "1.0.0-beta-2", "1.0.0-0"):
            self.assertEqual(release.release_version(tag), tag.removeprefix("v"))

    def test_invalid_tags(self):
        for tag in ("main", "v1.2", "v01.2.3", "1.2.3-01", "1.2.3-rc.01", "1.2.3+build", "1.2.3\n", "1.2.3;echo hello"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release.release_version(tag)

    def test_updates_manifests_and_lockfile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rust").mkdir()
            (root / "node").mkdir()
            cargo = root / "rust/Cargo.toml"
            cargo.write_text('[package]\nname = "clap-cpp"\nversion = "0.1.0"\n')
            (root / "node/package.json").write_text('{"name":"clap-cpp","version":"0.1.0"}')
            lock = {"version": "0.1.0", "packages": {"": {"version": "0.1.0"}, "node_modules/example": {"version": "2.0.0"}}}
            (root / "node/package-lock.json").write_text(json.dumps(lock))
            self.assertEqual(release.update(root, "v1.2.3-rc.1"), "1.2.3-rc.1")
            self.assertIn('version = "1.2.3-rc.1"', cargo.read_text())
            manifest = json.loads((root / "node/package.json").read_text())
            lock = json.loads((root / "node/package-lock.json").read_text())
            self.assertEqual(manifest["version"], "1.2.3-rc.1")
            self.assertEqual(lock["version"], manifest["version"])
            self.assertEqual(lock["packages"][""]["version"], manifest["version"])
            self.assertEqual(lock["packages"]["node_modules/example"]["version"], "2.0.0")


if __name__ == "__main__":
    unittest.main()
