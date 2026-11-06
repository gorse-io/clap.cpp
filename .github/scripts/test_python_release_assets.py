"""Execute the workflow's inline validator with UNIT FIXTURES, not release builds.

Set CLAP_TEST_CPU_WHEEL to inspect a real locally built CPU wheel as well.
Synthetic archives below contain labeled placeholder bytes, never native binaries;
passing these unit tests is not cross-platform build or release validation.
"""
import io
import os
import shutil
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
            archive.writestr(f"{directory}/{native}", (b"" if payload == "empty" else b"invalid" if payload == "invalid" else
                             (b"MZ" if platform == "win_amd64" else b"\xcf\xfa\xed\xfe" if platform.startswith("macosx_") else b"\x7fELF")
                             + b"UNIT FIXTURE, NOT NATIVE CODE"))
        if payload != "missing-notice":
            archive.writestr(f"{distribution}/licenses/clap-cpp.txt", "" if payload == "empty-notice" else "UNIT FIXTURE notice")
        if payload == "tests":
            archive.writestr(f"{distribution}/tests/test_leaked.py", "# TEST FIXTURE")
        if payload == "template":
            archive.writestr(f"{distribution}/unused.j2", "UNIT FIXTURE template")
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
            self.unit_sdist(source, distribution)
        (self.artifacts / MANIFEST).write_text("UNIT-FIXTURE-SONAME.so\n")

    def unit_sdist(self, source, distribution, omitted=None, extra=()):
        """Write wrapper-only UNIT FIXTURE source archives for the real heredoc."""
        with tarfile.open(source, "w:gz") as archive:
            for name in ("pyproject.toml", "README.md", "LICENSE",
                         *(f"{distribution}/{name}" for name in ("__init__.py", "clap.py", "clap.pyi", "py.typed")), *extra):
                if name == omitted:
                    continue
                data = b"# UNIT FIXTURE wrapper packaging input\n"
                entry = tarfile.TarInfo(f"{distribution}-0.1.0/{name}")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))

    def test_unit_fixture_sdist_requires_wrapper_and_rendered_metadata(self):
        for distribution in DISTRIBUTIONS:
            for required in ("pyproject.toml", "README.md", "LICENSE",
                             *(f"{distribution}/{name}" for name in ("__init__.py", "clap.py", "clap.pyi", "py.typed"))):
                with self.subTest(distribution=distribution, required=required):
                    source = self.artifacts / f"{distribution}-0.1.0.tar.gz"
                    self.unit_sdist(source, distribution, omitted=required)
                    self.assert_rejected("Sdist missing wrapper or packaging input")
                    self.unit_sdist(source, distribution)

    def test_unit_fixture_sdist_rejects_native_graph_and_binaries(self):
        for native in ("native/ggml/CMakeLists.txt", "CMakeLists.txt", "cmake/helpers.cmake", "src/clap.cpp",
                       "clap_cpp/libclap.so", "clap_cpp/clap.dll", "clap_cpp/libclap.dylib"):
            with self.subTest(native=native):
                self.unit_sdist(self.artifacts / "clap_cpp-0.1.0.tar.gz", "clap_cpp", extra=(native,))
                self.assert_rejected("Sdist contains native sources or compiled runtime")

    def test_published_archives_reject_tests(self):
        for distribution in DISTRIBUTIONS:
            with self.subTest(distribution=distribution):
                source = self.artifacts / f"{distribution}-0.1.0.tar.gz"
                self.unit_sdist(source, distribution, extra=("tests/test_leaked.py",))
                self.assert_rejected("Sdist contains tests")
                self.unit_sdist(source, distribution)
                unit_wheel(self.artifacts, distribution=distribution, payload="tests")
                self.assert_rejected("Wheel contains tests")
                unit_wheel(self.artifacts, distribution=distribution)

    def test_published_archives_reject_jinja_templates(self):
        for distribution in DISTRIBUTIONS:
            for template in ("pyproject.toml.j2", f"{distribution}/unused.j2"):
                with self.subTest(distribution=distribution, template=template):
                    source = self.artifacts / f"{distribution}-0.1.0.tar.gz"
                    self.unit_sdist(source, distribution, extra=(template,))
                    self.assert_rejected("Sdist contains Jinja template")
                    self.unit_sdist(source, distribution)
            with self.subTest(distribution=distribution, archive="wheel"):
                unit_wheel(self.artifacts, distribution=distribution, payload="template")
                self.assert_rejected("Wheel contains Jinja template")
                unit_wheel(self.artifacts, distribution=distribution)

    def run_validator(self):
        shutil.rmtree(self.root / "dist", ignore_errors=True)
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

    def test_unit_fixture_empty_or_invalid_native_rejected(self):
        for payload in ("empty", "invalid"):
            with self.subTest(payload=payload):
                self.replace_cpu(payload=payload)
                self.assert_rejected("Invalid package-local native library")

    def test_unit_fixture_runtime_notice_required(self):
        for payload in ("missing-notice", "empty-notice"):
            with self.subTest(payload=payload):
                self.replace_cpu(payload=payload)
                self.assert_rejected("Missing runtime notices")

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

    def test_actions_runtime_gates_reject_invalid_staging(self):
        # Execute all five real Actions heredocs, not a test copy of the checks.
        ci = WORKFLOW.with_name("ci.yml")
        cases = [(ci, "Package Python with the shared runtime", "build/python-package/clap_cpp"),
                 (WORKFLOW, "Copy external runtime into the CPU wrapper project", "python-dist/clap_cpp")]
        cases += [(WORKFLOW, f"Stage runtime and package canonical {backend.upper()} wrapper sdist",
                   f"runner/clap-{backend}/clap_cpp_{backend}") for backend in ("cuda", "vulkan", "rocm")]
        native = "clap.dll" if sys.platform == "win32" else "libclap.dylib" if sys.platform == "darwin" else "libclap.so"
        magic = b"MZ" if sys.platform == "win32" else b"\xcf\xfa\xed\xfe" if sys.platform == "darwin" else b"\x7fELF"
        for workflow, step_name, path in cases:
            step = workflow.read_text().split(f"- name: {step_name}\n", 1)[1]
            block = step.split("python - <<'PY'\n", 1)[1]
            source = textwrap.dedent(block.split("\n" + " " * (10 if workflow == ci else 8) + "PY\n", 1)[0])
            package = self.root / path
            package.mkdir(parents=True, exist_ok=True)
            runtime = package / native
            notice = package / "licenses/clap-cpp.txt"
            notice.parent.mkdir(exist_ok=True)
            for payload, license_text, error in ((None, "notice", "Missing staged external runtime"),
                                                (b"", "notice", "Missing staged external runtime"),
                                                (b"invalid", "notice", "Invalid staged external runtime"),
                                                (magic, None, "Missing runtime notices"),
                                                (magic, "  ", "Missing runtime notices"),
                                                (magic + b"UNIT FIXTURE, NOT NATIVE CODE", "notice", None)):
                with self.subTest(step=step_name, payload=payload, license_text=license_text):
                    runtime.unlink(missing_ok=True)
                    notice.unlink(missing_ok=True)
                    if payload is not None:
                        runtime.write_bytes(payload)
                    if license_text is not None:
                        notice.write_text(license_text)
                    result = subprocess.run([sys.executable, "-c", source], cwd=self.root,
                                            env=dict(os.environ, RUNNER_TEMP=str(self.root / "runner")),
                                            capture_output=True, text=True)
                    if error:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(error, result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)

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
