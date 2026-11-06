"""Package-only scikit-build-core wheels consume an explicitly copied external runtime."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

from jinja2 import Environment, StrictUndefined
import pytest

try:
    import tomllib
except ImportError:
    tomllib = pytest.importorskip("tomli")


PROJECT = Path(__file__).resolve().parents[1]
RUNTIME_NAME = "clap.dll" if sys.platform == "win32" else (
    "libclap.dylib" if sys.platform == "darwin" else "libclap.so"
)


def metadata(path=PROJECT):
    return tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))


def render(source, destination, backend="cpu", version=None):
    """Mirror the workflow's Jinja rendering and plain wrapper copies."""
    destination.mkdir(parents=True)
    package = "clap_cpp" + (f"_{backend}" if backend != "cpu" else "")
    for name in ("README.md", "LICENSE", "pyproject.toml.j2"):
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "clap_cpp", destination / package,
                    ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.dylib", "*.dll", "licenses"))
    if backend != "cpu":
        (destination / package / "__init__.py").write_text(
            f'"""Native {backend} runtime; import clap_cpp for the public API."""\n', encoding="utf-8")
    else:
        shutil.copytree(source / "tests", destination / "tests")
    template = Environment(undefined=StrictUndefined, keep_trailing_newline=True).from_string(
        (source / "pyproject.toml.j2").read_text(encoding="utf-8"))
    (destination / "pyproject.toml").write_text(template.render(
        backend=backend, version=version or metadata(source)["project"]["version"]), encoding="utf-8")
    return destination, package


def build(project, out, kind, env=None):
    return subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", f"--{kind}", "--outdir", str(out), str(project)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )


def test_template_matches_checked_in_cpu_metadata(tmp_path):
    project, _ = render(PROJECT, tmp_path / "cpu")
    assert (project / "pyproject.toml").read_bytes() == (PROJECT / "pyproject.toml").read_bytes()
    config = metadata(project)
    assert config["build-system"] == {"requires": ["scikit-build-core>=1.1.2"], "build-backend": "scikit_build_core.build"}
    settings = config["tool"]["scikit-build"]
    assert settings["wheel"]["cmake"] is False
    assert settings["wheel"]["platlib"] is True
    assert settings["wheel"]["py-api"] == "py3"
    assert settings["sdist"]["cmake"] is False
    assert settings["sdist"]["inclusion-mode"] == "explicit"
    assert settings["wheel"]["force-include"] == {"clap_cpp": "clap_cpp"}


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm", "vulkan"])
def test_wheel_and_wrapper_only_sdist(tmp_path, backend):
    project, package = render(PROJECT, tmp_path / "project", backend, "2.3.4rc1")
    config = metadata(project)
    (project / ".gitignore").write_text("*.so\n*.dll\n*.dylib\ncompanions/\nlicenses/\n")
    (project / "unlisted.txt").write_text("Must not enter explicit sdists")
    (project / "tests").mkdir(exist_ok=True)
    (project / "tests/test_not_for_distribution.py").write_text("# Source-only test fixture")
    (project / package / "tests").mkdir()
    (project / package / "tests/test_not_for_distribution.py").write_text("# Must not leak via force inclusion")
    (project / package / "__pycache__").mkdir()
    (project / package / "__pycache__/clap.pyc").write_bytes(b"cache fixture")
    assert config["project"]["name"] == package.replace("_", "-")
    assert config["project"]["version"] == "2.3.4rc1"
    assert config["project"]["dependencies"] == []
    if backend == "cpu":
        for name in ("cuda", "rocm", "vulkan"):
            assert config["project"]["optional-dependencies"][name] == [f"clap-cpp-{name}==2.3.4rc1"]
    for name in ("clap.py", "clap.pyi", "py.typed"):
        assert (project / package / name).read_bytes() == (PROJECT / "clap_cpp" / name).read_bytes()
    # Assembly must not invoke a native build, including when building from sdist.
    tools = tmp_path / "tools"
    tools.mkdir()
    marker = tmp_path / "compiler-called"
    for tool in ("cmake", "ninja", "cc", "c++", "gcc", "g++", "clang", "clang++", "cl"):
        executable = tools / tool
        executable.write_text(f'#!/bin/sh\nprintf called > "{marker}"\nexit 91\n')
        executable.chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""))
    out = tmp_path / "dist"
    metadata_dir = project / "metadata"
    metadata_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", "from scikit_build_core.build import prepare_metadata_for_build_wheel; "
         "prepare_metadata_for_build_wheel('metadata')"],
        cwd=project, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(metadata_dir.glob("*.dist-info/METADATA"))
    # Validation belongs to Actions, not a replacement Python backend/hook.
    # Prefer an actual externally compiled runtime; fixture-only jobs still test copying.
    native = project / package / RUNTIME_NAME
    external = os.environ.get("CLAP_TEST_EXTERNAL_RUNTIME")
    if external:
        shutil.copy2(Path(external) / RUNTIME_NAME, native)
    else:
        native.write_bytes(b"packaging fixture only, NOT an executable runtime")
    notices = project / package / "licenses"
    notices.mkdir()
    shutil.copy2(project / "LICENSE", notices / "clap-cpp.txt")
    (notices / "dependency.txt").write_text("Dependency notice")
    companion = project / package / "companions" / "libdependency.so.1"
    companion.parent.mkdir()
    companion.write_bytes(b"companion fixture")
    result = build(project, out, "wheel", env)
    assert result.returncode == 0, result.stdout
    wheel = next(out.glob("*.whl"))
    assert "-py3-none-" in wheel.name and not wheel.name.endswith("-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.namelist()
        info = archive.read(next(n for n in entries if n.endswith(".dist-info/WHEEL")))
        assert b"Root-Is-Purelib: false" in info
        assert archive.read(f"{package}/{RUNTIME_NAME}") == native.read_bytes()
        assert archive.read(f"{package}/companions/libdependency.so.1") == companion.read_bytes()
        assert archive.read(f"{package}/licenses/clap-cpp.txt") == (project / "LICENSE").read_bytes()
        assert f"{package}/licenses/dependency.txt" in entries
        for name in ("__init__.py", "clap.py", "clap.pyi", "py.typed"):
            assert f"{package}/{name}" in entries
        assert not any("__pycache__" in n or "tests" in Path(n).parts for n in entries)
    result = build(project, out, "sdist", env)
    assert result.returncode == 0, result.stdout
    with tarfile.open(next(out.glob("*.tar.gz"))) as archive:
        entries = {n.split("/", 1)[1] for n in archive.getnames() if "/" in n}
        required = {"pyproject.toml", "pyproject.toml.j2", "README.md", "LICENSE",
                    *(f"{package}/{name}" for name in ("__init__.py", "clap.py", "clap.pyi", "py.typed"))}
        assert required <= entries
        assert not any(n.startswith("native/") or "licenses/" in n or "companions/" in n or
                       n.endswith((".so", ".dll", ".dylib", ".cpp")) for n in entries)
        assert not any("__pycache__" in n or n.endswith(".pyc") for n in entries)
        assert "unlisted.txt" not in entries
        assert not any("tests" in Path(n).parts for n in entries)
        assert not {"stage_runtime.py", "prepare_backend.py", "_build_backend.py", "hatch_build.py"} & entries
        archive.extractall(tmp_path / "extracted", filter="data")
    source = next((tmp_path / "extracted").iterdir())
    shutil.copytree(project / package, source / package, dirs_exist_ok=True)
    result = build(source, tmp_path / "from-sdist", "wheel", env)
    assert result.returncode == 0, result.stdout
    if backend == "cpu":
        render(source, tmp_path / "cuda-from-cpu-sdist", "cuda")
    assert not marker.exists()


def test_editable_is_platform_tagged_without_compilation(tmp_path):
    project, _ = render(PROJECT, tmp_path / "project")
    result = subprocess.run(
        [sys.executable, "-c", "from scikit_build_core.build import build_editable; build_editable('editable')"],
        cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((project / "editable").glob("*.whl"))
    assert "-py3-none-" in wheel.name and not wheel.name.endswith("-any.whl")


def test_macos_native_tag_and_minimum(monkeypatch):
    from scikit_build_core.builder import wheel_tag
    import sysconfig
    sysconfig.get_config_vars()  # Initialize real host configuration before mocking Darwin.
    monkeypatch.setattr(wheel_tag.sys, "platform", "darwin")
    # packaging uses the executing architecture, not a universal2 interpreter tag.
    monkeypatch.setattr(wheel_tag.packaging.tags.platform, "mac_ver", lambda: ("15.0", ("", "", ""), "arm64"))
    tag = wheel_tag.WheelTag.compute_best([], "py3", env={"MACOSX_DEPLOYMENT_TARGET": "11.0"})
    assert str(tag) == "py3-none-macosx_11_0_arm64"
