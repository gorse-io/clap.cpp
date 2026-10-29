import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("verify_crate", Path(__file__).with_name("verify-crate.py"))
assert spec is not None and spec.loader is not None
crate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crate)


class CrateSourcesTest(unittest.TestCase):
    def archive(self, path, files):
        with tarfile.open(path, "w:gz") as archive:
            for name, content in files.items():
                content = content.encode()
                member = tarfile.TarInfo("clap-cpp-0.1.0/" + name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

    def test_repackaging_metadata_only(self):
        files = {"Cargo.toml": "normalized", "vendor/include/clap.h": "native",
                 "Cargo.toml.orig": "original", ".cargo_vcs_info.json": "revision"}
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.crate", Path(directory) / "b.crate"
            self.archive(a, files)
            files["Cargo.toml.orig"] = "normalized"
            del files[".cargo_vcs_info.json"]
            self.archive(b, files)
            crate.verify(a, b)
            for name in ("Cargo.toml", "vendor/include/clap.h"):
                changed = dict(files, **{name: "modified"})
                self.archive(b, changed)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    crate.verify(a, b)
            files["unexpected.cpp"] = "new"
            self.archive(b, files)
            with self.assertRaises(ValueError):
                crate.verify(a, b)

    def test_missing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.crate"
            self.archive(path, {"Cargo.toml": "manifest"})
            with self.assertRaises(ValueError):
                crate.sources(path)


if __name__ == "__main__":
    unittest.main()
