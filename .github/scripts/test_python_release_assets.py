"""Execute the workflow's inline validator with UNIT FIXTURES, not release builds.

Set CLAP_TEST_CPU_WHEEL to inspect a real locally built CPU wheel as well.
Synthetic archives below contain labeled placeholder bytes, never native binaries;
passing these unit tests is not cross-platform build or release validation.
"""
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
import zipfile

WORKFLOW = Path(__file__).parents[1] / "workflows/release.yml"
PLATFORMS = ("manylinux_2_35_x86_64", "manylinux_2_35_aarch64", "macosx_11_0_arm64", "win_amd64")
DISTRIBUTIONS = ("clap_cpp", "clap_cpp_cuda", "clap_cpp_rocm", "clap_cpp_vulkan")
MANIFEST = "ROCM-6.4.3-EXCLUDED-SONAMES.txt"


def validator_source():
    step = WORKFLOW.read_text().split("    - name: Validate complete asset set and write runtime manifest\n", 1)[1]
    return textwrap.dedent(step.split("        python - <<'PY'\n", 1)[1].split("        PY\n", 1)[0])


def unit_wheel(root, distribution="clap_cpp", platform=PLATFORMS[0], interpreter="py3", abi="none", payload="correct"):
    """Write a UNIT FIXTURE ZIP, deliberately not a buildable/installable wheel."""
    path = root / f"{distribution}-0.1.0-{interpreter}-{abi}-{platform}.whl"
    native = "clap.dll" if platform == "win_amd64" else "libclap.dylib" if platform.startswith("macosx_") else "libclap.so"
    with zipfile.ZipFile(path, "w") as archive:
        if payload != "missing-wrapper":
            archive.writestr(f"{distribution}/clap.py", "# UNIT FIXTURE ctypes wrapper, not runnable\n")
        if payload == "extension":
            archive.writestr(f"{distribution}/clap.cpython-311-x86_64-linux-gnu.so", b"UNIT FIXTURE, NOT NATIVE CODE")
        else:
            directory = "wrong_package" if payload == "wrong-package" else distribution
            archive.writestr(f"{directory}/{native}", b"UNIT FIXTURE, NOT NATIVE CODE")
    return path


class PythonReleaseAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        for distribution in DISTRIBUTIONS:
            for platform in PLATFORMS if distribution == "clap_cpp" else PLATFORMS[:1]:
                unit_wheel(self.artifacts, distribution, platform)
            source = self.artifacts / f"{distribution}-0.1.0.tar.gz"
            with tarfile.open(source, "w:gz") as archive:
                data = b"# UNIT FIXTURE pinned sources\n"
                entry = tarfile.TarInfo(f"{distribution}-0.1.0/native/ggml/CMakeLists.txt")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        (self.artifacts / MANIFEST).write_text("UNIT-FIXTURE-SONAME.so\n")

    def run_validator(self):
        return subprocess.run([sys.executable, "-c", validator_source()], cwd=self.root, capture_output=True, text=True)

    def replace_cpu(self, **kwargs):
        next(self.artifacts.glob("clap_cpp-*.whl")).unlink()
        return unit_wheel(self.artifacts, **kwargs)

    def assert_rejected(self, diagnostic):
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(diagnostic, result.stderr)

    def test_unit_fixture_complete_ctypes_matrix(self):
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validated 7 wheels and 4 sdists", result.stdout)
        runtime = (self.root / "dist/PYTHON-WHEEL-RUNTIME.txt").read_text()
        self.assertIn("py3-none", runtime)
        self.assertNotIn("matching CPython/platform tags", runtime)
        self.assertEqual(len(list((self.root / "dist").glob("*.whl"))), 7)

    def test_unit_fixture_cpython_interpreter_rejected(self):
        self.replace_cpu(interpreter="cp311")
        self.assert_rejected("Python/ABI tags")

    def test_unit_fixture_non_none_abi_rejected(self):
        self.replace_cpu(abi="abi3")
        self.assert_rejected("Python/ABI tags")

    def test_unit_fixture_old_extension_rejected(self):
        self.replace_cpu(payload="extension")
        self.assert_rejected("package-local native library")

    def test_unit_fixture_missing_wrapper_rejected(self):
        self.replace_cpu(payload="missing-wrapper")
        self.assert_rejected("ctypes wrapper")

    def test_unit_fixture_native_in_wrong_package_rejected(self):
        self.replace_cpu(payload="wrong-package")
        self.assert_rejected("package-local native library")

    def test_unit_fixture_unexpected_architecture_rejected(self):
        self.replace_cpu(platform="manylinux_2_35_ppc64le")
        self.assert_rejected("Unexpected platform")

    def test_unit_fixture_missing_platform_rejected(self):
        next(self.artifacts.glob("*win_amd64.whl")).unlink()
        self.assert_rejected("Wheel matrix mismatch")

    def test_unit_fixture_duplicate_platform_rejected(self):
        duplicate = self.artifacts / "duplicate"
        duplicate.mkdir()
        unit_wheel(duplicate)
        self.assert_rejected("Duplicate release asset")

    def test_unit_fixture_single_rocm_manifest_required(self):
        (self.artifacts / MANIFEST).unlink()
        self.assert_rejected("ROCm exclusion manifest")

    def test_unit_fixture_six_rocm_manifests_rejected(self):
        for index in range(5):
            directory = self.artifacts / str(index)
            directory.mkdir()
            (directory / MANIFEST).write_text("UNIT-FIXTURE-SONAME.so\n")
        self.assert_rejected("ROCm exclusion manifest")

    @unittest.skipUnless(os.environ.get("CLAP_TEST_CPU_WHEEL"), "set CLAP_TEST_CPU_WHEEL for real built wheel inspection")
    def test_actual_built_cpu_payload_and_unrepaired_release_rejection(self):
        from packaging.utils import parse_wheel_filename
        wheel = Path(os.environ["CLAP_TEST_CPU_WHEEL"])
        name, _, _, tags = parse_wheel_filename(wheel.name)
        self.assertEqual(name, "clap-cpp")
        self.assertTrue(all(tag.interpreter == "py3" and tag.abi == "none" and tag.platform.startswith("linux_") for tag in tags))
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            wrapper = archive.read("clap_cpp/clap.py")
            self.assertIn(b"C.CDLL", wrapper)
            self.assertTrue(archive.read("clap_cpp/libclap.so").startswith(b"\x7fELF"))
            self.assertFalse(any(Path(entry).name.startswith("clap.") and entry.endswith((".so", ".pyd")) for entry in members))
            metadata = archive.read(next(entry for entry in members if entry.endswith(".dist-info/WHEEL")))
            self.assertIn(b"Root-Is-Purelib: false", metadata)
        # Real bytes and original filename: never relabel a raw Linux build as manylinux.
        (self.artifacts / wheel.name).write_bytes(wheel.read_bytes())
        self.assert_rejected("Unrepaired Linux wheel")


if __name__ == "__main__":
    unittest.main()
