"""CUDA release archive fixtures; these do not build or impersonate CUDA binaries.

Run with: python -m unittest discover -s tests -p test_cuda_release_packaging.py -v
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/package_cuda_release.py"
PRESETS = json.loads((ROOT / "CMakePresets.json").read_text())["configurePresets"]
CUDA_PRESETS = [p for p in PRESETS if "-cuda" in p["name"]]


class CUDAReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cuda-release-", dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cuda = self.root / "toolkit with spaces"
        self.output = self.root / "archives"
        self.binary = self.root / "clap fixture"
        self.binary.write_text("Executable packaging fixture, not a CUDA binary.\n")
        self.binary.chmod(0o755)

    def package(self, platform, arch, asset, success=True):
        result = subprocess.run([
            __import__("sys").executable, str(SCRIPT), "--platform", platform,
            "--cuda-root", str(self.cuda), "--cuda-arch", arch,
            "--binary", str(self.binary), "--asset", asset,
            "--output-dir", str(self.output),
        ], text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def linux_libraries(self, arch="x86_64"):
        lib = self.cuda / "targets" / f"{arch}-linux" / "lib"
        lib.mkdir(parents=True)
        for name, version in (("cudart", "12"), ("cublas", "12"), ("cublasLt", "12")):
            real = lib / f"lib{name}.so.{version}.9.1"
            real.write_text(f"{name} archive fixture, not a CUDA library.\n")
            (lib / f"lib{name}.so.{version}").symlink_to(real.name)
            (lib / f"lib{name}.so").symlink_to(f"lib{name}.so.{version}")
        (lib / "libcuda.so.1").write_text("Driver must not be shipped\n")
        (lib / "libcudart_static.a").write_text("Static archive must not be shipped\n")
        return lib

    def test_linux_archives_are_separate_with_relocatable_soname_links(self):
        for arch in ("x86_64", "aarch64"):
            with self.subTest(arch=arch):
                self.linux_libraries(arch)
                asset = f"clap-linux-{arch}-cuda12.9.tar.gz"
                self.package("linux", arch, asset)
                with tarfile.open(self.output / asset) as archive:
                    self.assertEqual(archive.getnames(), ["clap"])
                    self.assertTrue(archive.getmember("clap").mode & 0o111)
                with tarfile.open(self.output / f"cudart-{asset}") as archive:
                    entries = {m.name: m for m in archive}
                    self.assertEqual(len(entries), 6)
                    for name in ("cudart", "cublas", "cublasLt"):
                        alias = entries[f"lib{name}.so.12"]
                        self.assertTrue(alias.issym())
                        self.assertEqual(alias.linkname, f"lib{name}.so.12.9.1")
                        self.assertTrue(entries[alias.linkname].isfile())
                    self.assertFalse(any("libcuda.so" in name or name.endswith(".a") for name in entries))

    def test_windows_archives_are_separate_and_arch_specific(self):
        for arch in ("x64", "arm64"):
            with self.subTest(arch=arch):
                lib = self.cuda / "bin" / arch
                lib.mkdir(parents=True)
                expected = {"cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"}
                for name in expected | {"nvcuda.dll", "nvcc.exe"}:
                    (lib / name).write_text(f"{arch} {name} archive fixture, not a DLL.\n")
                asset = f"clap-windows-{arch}-cuda13.4.zip"
                self.package("windows", arch, asset)
                with zipfile.ZipFile(self.output / asset) as archive:
                    self.assertEqual(archive.namelist(), ["clap.exe"])
                with zipfile.ZipFile(self.output / f"cudart-{asset}") as archive:
                    self.assertEqual(set(archive.namelist()), expected)
                    self.assertTrue(all(archive.read(name).startswith(arch.encode()) for name in expected))

    def test_missing_runtime_fails_before_publishing_either_archive(self):
        lib = self.linux_libraries()
        for path in lib.glob("libcublasLt.so*"):
            path.unlink()
        result = self.package("linux", "x86_64", "clap.tar.gz", success=False)
        self.assertIn("cublasLt", result.stderr)
        self.assertFalse(list(self.output.glob("*")))

    def test_windows_missing_cudart_is_not_silently_accepted(self):
        lib = self.cuda / "bin" / "arm64"
        lib.mkdir(parents=True)
        for name in ("cublas64_13.dll", "cublasLt64_13.dll"):
            (lib / name).write_text("Archive fixture\n")
        result = self.package("windows", "arm64", "clap.zip", success=False)
        self.assertIn("cudart", result.stderr)
        self.assertFalse(list(self.output.glob("*")))

    def test_presets_keep_static_backends_but_dynamic_cuda(self):
        self.assertEqual(len(CUDA_PRESETS), 4)
        base = next(p for p in PRESETS if p["name"] == "release-base")["cacheVariables"]
        self.assertEqual(base["BUILD_SHARED_LIBS"], "OFF")
        self.assertEqual(base["GGML_BACKEND_DL"], "OFF")
        for preset in CUDA_PRESETS:
            with self.subTest(preset=preset["name"]):
                variables = preset["cacheVariables"]
                self.assertEqual(variables["GGML_STATIC"], "OFF")
                self.assertEqual(variables["CMAKE_CUDA_RUNTIME_LIBRARY"], "Shared")
                self.assertEqual(variables["GGML_CUDA"], "ON")
                expected = "75-virtual;80-virtual;86-real;89-real;90;120a-real;121a-real"
                if preset["name"].startswith("linux"):
                    expected = "50-virtual;61-virtual;70-virtual;" + expected
                    self.assertEqual(variables["CMAKE_INSTALL_RPATH"], "$ORIGIN")
                    self.assertEqual(variables["CMAKE_BUILD_WITH_INSTALL_RPATH"], "ON")
                    self.assertEqual(variables["CMAKE_INSTALL_RPATH_USE_LINK_PATH"], "OFF")
                self.assertEqual(variables["CMAKE_CUDA_ARCHITECTURES"], expected)

    @unittest.skipUnless(shutil.which("cc") and shutil.which("cmake") and shutil.which("readelf"),
                         "requires Linux C compiler, CMake and readelf")
    def test_linux_preset_rpath_relocates_real_non_cuda_executable(self):
        """Exercise the preset's linker settings, not a pretend CUDA build."""
        variables = next(p for p in CUDA_PRESETS if p["name"].startswith("linux"))["cacheVariables"]
        self.assertEqual(variables.get("CMAKE_INSTALL_RPATH"), "$ORIGIN")
        source = self.root / "source"
        source.mkdir()
        (source / "leaf.c").write_text("int leaf(void) { return 42; }\n")
        (source / "middle.c").write_text("int leaf(void); int middle(void) { return leaf(); }\n")
        (source / "main.c").write_text("int middle(void); int main(void) { return middle() != 42; }\n")
        (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(rpath_fixture C)
add_library(fixture_leaf SHARED leaf.c)
add_library(fixture_middle SHARED middle.c)
target_link_libraries(fixture_middle PRIVATE fixture_leaf)
# Only the executable gets the preset RPATH, to test transitive lookup too.
set_target_properties(fixture_leaf fixture_middle PROPERTIES SKIP_BUILD_RPATH YES)
add_executable(fixture main.c)
target_link_libraries(fixture PRIVATE fixture_middle)
''')
        build = self.root / "build"
        options = [f"-D{k}={v}" for k, v in variables.items() if k.startswith("CMAKE_") and k != "CMAKE_CUDA_ARCHITECTURES"]
        subprocess.run(["cmake", "-S", str(source), "-B", str(build), *options], check=True, capture_output=True)
        subprocess.run(["cmake", "--build", str(build)], check=True, capture_output=True)
        dynamic = subprocess.check_output(["readelf", "-d", str(build / "fixture")], text=True)
        self.assertIn("(RPATH)", dynamic)
        self.assertIn("[$ORIGIN]", dynamic)
        relocated = self.root / "relocated"
        relocated.mkdir()
        for name in ("fixture", "libfixture_middle.so", "libfixture_leaf.so"):
            shutil.move(str(build / name), relocated / name)
        shutil.rmtree(build)
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)
        subprocess.run([str(relocated / "fixture")], env=env, cwd=self.root, check=True)


if __name__ == "__main__":
    unittest.main()
