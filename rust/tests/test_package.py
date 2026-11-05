"""Check the real Cargo inventory in checkouts and extracted source packages."""
from pathlib import Path, PureWindowsPath
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULES = ("ggml", "thirdparty/json", "thirdparty/icu", "thirdparty/miniaudio")
REQUIRED = (
    "Cargo.toml", "Cargo.lock", "LICENSE", "CMakeLists.txt",
    "cmake/cargo-link.cmake", "cmake/static-development.cmake", "include/clap.h",
    "rust/build.rs", "rust/src/lib.rs", "rust/examples/embed.rs",
    "rust/tests/bindings.rs", "rust/tests/ffi/regression.rs",
    "rust/tests/generate.py", "rust/tests/test_package.py", "rust/README.md",
    "python/tests/tiny_models.py", "ggml/CMakeLists.txt", "ggml/LICENSE",
    "ggml/src/ggml-cuda/CMakeLists.txt", "ggml/src/ggml-vulkan/CMakeLists.txt",
    "ggml/src/ggml-hip/CMakeLists.txt", "ggml/src/ggml-metal/CMakeLists.txt",
    "thirdparty/json/CMakeLists.txt", "thirdparty/json/LICENSE.MIT",
    "thirdparty/miniaudio/miniaudio.h", "thirdparty/miniaudio/LICENSE",
    "thirdparty/icu/LICENSE", "thirdparty/icu/icu4c/source/common/unicode/uvernum.h",
    "thirdparty/icu/icu4c/source/i18n/regexcmp.cpp",
    "thirdparty/icu/icu4c/source/stubdata/stubdata.cpp",
)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def package_paths(output):
    # Cargo --list displays host-native separators, while archive keys are POSIX.
    return {PureWindowsPath(name).as_posix() for name in output.splitlines()}


class PackageTests(unittest.TestCase):
    def test_inventory_paths_are_platform_independent(self):
        expected = {"Cargo.toml", "rust/src/lib.rs", "thirdparty/icu/LICENSE"}
        self.assertEqual(package_paths("Cargo.toml\nrust/src/lib.rs\nthirdparty/icu/LICENSE\n"), expected)
        self.assertEqual(package_paths("Cargo.toml\r\nrust\\src\\lib.rs\r\nthirdparty\\icu\\LICENSE\r\n"), expected)

    def test_root_package_inventory(self):
        files = package_paths(subprocess.check_output(
            ["cargo", "package", "--manifest-path", str(ROOT / "Cargo.toml"),
             "--list", "--allow-dirty", "--offline"], cwd=ROOT, text=True))
        self.assertFalse(set(REQUIRED) - files, f"missing: {set(REQUIRED) - files}")
        for name in files:
            parts = Path(name).parts
            self.assertFalse(set(parts) & {"target", "vendor", ".git", "__pycache__"}, name)
            self.assertNotIn(Path(name).suffix, (".gguf", ".wav", ".pyc", ".o", ".a"), name)
        self.assertFalse(any(name.startswith(("thirdparty/CLI11/", "thirdparty/cpp-httplib/"))
                             for name in files))
        # Every original native source/module and selected dependency file belongs
        # to the package, not a generated copy. Check all files, not just sentinels.
        for tree in ("src", "include", "cmake", "ggml/src", "ggml/include", "ggml/cmake",
                     "thirdparty/json/include", "thirdparty/json/cmake",
                     "thirdparty/icu/icu4c/source/common",
                     "thirdparty/icu/icu4c/source/i18n", "thirdparty/icu/icu4c/source/stubdata"):
            expected = {p.relative_to(ROOT).as_posix() for p in (ROOT / tree).rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts}
            self.assertTrue(expected, f"empty source tree: {tree}")
            self.assertFalse(expected - files, f"omitted {tree}: {expected - files}")

    def test_checkout_submodules_are_pinned_and_clean(self):
        if not (ROOT / ".git").exists():
            self.skipTest("registry source archive has no Git metadata")
        for module in MODULES:
            with self.subTest(module=module):
                entry = git("ls-tree", "HEAD", "--", module).split()
                self.assertEqual(entry[:2], ["160000", "commit"])
                directory = ROOT / module
                self.assertTrue((directory / ".git").exists(), "initialize pinned submodules")
                self.assertEqual(git("rev-parse", "HEAD", cwd=directory), entry[2])
                self.assertEqual(git("status", "--porcelain", "--untracked-files=all", cwd=directory), "")


if __name__ == "__main__":
    unittest.main()
