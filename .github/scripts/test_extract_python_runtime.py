"""Extraction unit tests: archives contain labeled bytes, NOT compiled binaries."""
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).with_name("extract-python-runtime.py")
PLATFORMS = {
    "linux-x64": "manylinux_2_28_x86_64.manylinux2014_x86_64",
    "linux-arm64": "manylinux_2_28_aarch64",
    "darwin-arm64": "macosx_11_0_arm64",
    "win32-x64": "win_amd64",
}
LIBRARIES = {"linux-x64": "libclap.so", "linux-arm64": "libclap.so",
             "darwin-arm64": "libclap.dylib", "win32-x64": "clap.dll"}
LICENSES = ("ggml", "miniaudio", "icu", "json")
UNIT_BYTES = b"UNIT FIXTURE ONLY; NOT A COMPILED SHARED LIBRARY\x00\xff\r\n"


class ExtractPythonRuntimeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.wheels = self.root / "wheels"
        self.wheels.mkdir()
        self.output = self.root / "runtime"

    def wheel(self, target="linux-x64", platform=None, version="1.2.3", distribution="clap_cpp",
              interpreter="py3", abi="none", payload=None, extra=None, directory=None):
        directory = directory or self.wheels
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{distribution}-{version}-{interpreter}-{abi}-{platform or PLATFORMS[target]}.whl"
        files = {f"clap_cpp/{LIBRARIES[target]}": UNIT_BYTES,
                 "clap_cpp/clap.py": b"# UNIT FIXTURE Python wrapper\n",
                 "clap_cpp/clap.pyi": b"# UNIT FIXTURE stub\n",
                 "clap_cpp/py.typed": b"",
                 f"clap_cpp-{version}.dist-info/METADATA": b"UNIT FIXTURE metadata",
                 f"clap_cpp-{version}.dist-info/licenses/LICENSE": b"UNIT FIXTURE core license"}
        files.update({f"clap_cpp/licenses/{license}.txt": f"UNIT FIXTURE {license}".encode()
                      for license in LICENSES})
        if payload is not None:
            files = payload
        files.update(extra or {})
        path = directory / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, data in files.items():
                archive.writestr(member, data)
        return path

    def run_cli(self, target: str | None = "linux-x64", version=None):
        command = [sys.executable, str(SCRIPT), "--wheels", str(self.wheels), "--output", str(self.output)]
        if target:
            command.extend(["--target", target])
        if version:
            command.extend(["--version", version])
        return subprocess.run(command, capture_output=True, text=True)

    def rejected(self, diagnostic, **kwargs):
        result = self.run_cli(**kwargs)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(diagnostic, result.stderr)

    def test_default_recursively_extracts_all_four_targets_without_python(self):
        for target in PLATFORMS:
            self.wheel(target, directory=self.wheels / target / "nested")
        result = self.run_cli(target=None, version="1.2.3")
        self.assertEqual(result.returncode, 0, result.stderr)
        for target, library in LIBRARIES.items():
            runtime = self.output / target
            self.assertEqual((runtime / "clap_cpp" / library).read_bytes(), UNIT_BYTES)
            self.assertEqual((runtime / "clap_cpp/licenses/clap.txt").read_bytes(), b"UNIT FIXTURE core license")
            self.assertFalse(any(path.suffix in {".py", ".pyi"} or ".dist-info" in str(path)
                                 or path.name == "py.typed" for path in runtime.rglob("*")))

    def test_ordinary_linux_tags_and_single_target(self):
        self.wheel(platform="linux_x86_64")
        self.wheel("linux-arm64", platform="linux_aarch64")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.output / "linux-arm64").exists())

    def test_pep440_prerelease_matches_npm_semver(self):
        self.wheel(version="1.2.3rc2")
        result = self.run_cli(version="1.2.3-rc.2")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_version_mismatch(self):
        self.wheel()
        self.rejected("version mismatch", version="2.0.0")

    def test_missing_requested_targets(self):
        self.wheel()
        self.rejected("Missing", target=None)

    def test_duplicate_target(self):
        self.wheel()
        self.wheel(platform="linux_x86_64", directory=self.wheels / "duplicate")
        self.rejected("Duplicate")

    def test_ignores_known_gpu_wheels(self):
        self.wheel()
        self.wheel(distribution="clap_cpp_cuda")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unexpected_distribution(self):
        self.wheel(distribution="another_project")
        self.rejected("distribution")

    def test_unsupported_platform_and_python_abi(self):
        for kwargs in ({"platform": "manylinux_2_28_ppc64le"}, {"platform": "musllinux_1_2_x86_64"},
                       {"platform": "macosx_11_0_x86_64"}, {"platform": "win_arm64"},
                       {"interpreter": "cp311"}, {"abi": "abi3"},
                       {"platform": "linux_x86_64.linux_aarch64"}):
            with self.subTest(kwargs=kwargs):
                path = self.wheel(**kwargs)
                self.rejected("Unsupported")
                path.unlink()

    def test_incorrect_core_payload_names(self):
        for name in ("wrong/libclap.so", "clap_cpp/libclap.dylib", "clap_cpp/clap.cpython-311.so"):
            with self.subTest(name=name):
                self.wheel(payload={name: UNIT_BYTES})
                self.rejected("core library")

    def test_required_dependency_licenses(self):
        path = self.wheel()
        with zipfile.ZipFile(path) as archive:
            payload = {name: archive.read(name) for name in archive.namelist()
                       if name != "clap_cpp/licenses/icu.txt"}
        self.wheel(payload=payload)
        self.rejected("license")

    def test_core_license_optional_and_companions_preserve_bytes_and_paths(self):
        path = self.wheel(extra={"clap_cpp.libs/libdep-123.so.1": UNIT_BYTES,
                                 "clap_cpp/libs/libother.so": UNIT_BYTES,
                                 "clap_cpp.libs/NOTICE.txt": b"UNIT dependency notice"})
        with zipfile.ZipFile(path) as archive:
            payload = {name: archive.read(name) for name in archive.namelist() if ".dist-info/" not in name}
        self.wheel(payload=payload)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        for name, data in payload.items():
            if name.endswith(".so") or ".so." in name or "NOTICE" in name:
                self.assertEqual((self.output / "linux-x64" / name).read_bytes(), data)

    def test_macos_nested_repaired_companion(self):
        self.wheel("darwin-arm64", extra={"clap_cpp/.dylibs/libdependency.dylib": UNIT_BYTES})
        result = self.run_cli(target="darwin-arm64")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.output / "darwin-arm64/clap_cpp/.dylibs/libdependency.dylib").read_bytes(), UNIT_BYTES)

    def test_windows_adjacent_dll_preserved(self):
        self.wheel("win32-x64", extra={"clap_cpp/dependency.dll": UNIT_BYTES})
        result = self.run_cli(target="win32-x64")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.output / "win32-x64/clap_cpp/dependency.dll").read_bytes(), UNIT_BYTES)

    def test_windows_delvewheel_search_path_cannot_be_silently_lost(self):
        self.wheel("win32-x64", extra={"clap_cpp.libs/dependency.dll": UNIT_BYTES})
        self.rejected("Windows", target="win32-x64")

    def test_unknown_native_location_rejected(self):
        self.wheel(extra={"other.libs/libdependency.so": UNIT_BYTES})
        self.rejected("native companion")

    def test_unsafe_archive_paths_even_in_unselected_metadata(self):
        for name in ("../escape", "/absolute", "clap_cpp/../escape", "clap_cpp\\evil.dll",
                     "C:/evil", "clap_cpp/a:stream", "clap_cpp//evil", "clap_cpp/CON",
                     "clap_cpp/trailing.", "clap_cpp/./evil"):
            with self.subTest(name=name):
                self.wheel(extra={name: b"UNIT UNSAFE PATH"})
                self.rejected("Unsafe")

    def test_symlink_entry_rejected(self):
        path = self.wheel()
        with zipfile.ZipFile(path, "a") as archive:
            info = zipfile.ZipInfo("clap_cpp/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../../escape")
        self.rejected("symlink")

    def test_python_files_never_copied_even_in_native_license_directory(self):
        self.wheel(extra={"clap_cpp/licenses/helper.py": b"# UNIT FIXTURE Python code"})
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.output / "linux-x64/clap_cpp/licenses/helper.py").exists())

    def test_case_colliding_paths_rejected(self):
        self.wheel(extra={"clap_cpp/LIBCLAP.SO": UNIT_BYTES})
        self.rejected("colliding")

    def test_archive_file_directory_collision_rejected(self):
        self.wheel(extra={"clap_cpp/libs": b"UNIT regular file",
                          "clap_cpp/libs/libdep.so": UNIT_BYTES})
        self.rejected("collision")

    def test_output_symlink_rejected(self):
        actual = self.root / "actual"
        actual.mkdir()
        try:
            self.output.symlink_to(actual, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Cannot create symlink on this host: {error}")
        self.wheel()
        self.rejected("symlink output")
        self.assertEqual(list(actual.iterdir()), [])

    def test_invalid_version_is_clear_error(self):
        self.wheel()
        self.rejected("Invalid version", version="not-semver")

    def test_all_target_validation_finishes_before_publishing_any(self):
        for target in PLATFORMS:
            self.wheel(target, extra={"../escape": b"UNIT invalid"} if target == "win32-x64" else None)
        self.rejected("Unsafe", target=None)
        self.assertFalse(any((self.output / target).exists() for target in PLATFORMS))

    def test_mixed_versions_rejected_without_explicit_version(self):
        for target in PLATFORMS:
            self.wheel(target, version="1.2.4" if target == "win32-x64" else "1.2.3")
        self.rejected("version mismatch", target=None)

    def test_existing_target_refused_without_modification(self):
        self.wheel()
        target = self.output / "linux-x64"
        target.mkdir(parents=True)
        (target / "sentinel").write_bytes(b"keep")
        self.rejected("already exists")
        self.assertEqual((target / "sentinel").read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
