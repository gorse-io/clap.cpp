"""Package-only Hatchling wheels consume an explicitly copied external runtime."""

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
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
    for name in ("README.md", "LICENSE", "hatch_build.py", "pyproject.toml.j2"):
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


@pytest.fixture
def macos_hook(tmp_path, monkeypatch):
    module = runpy.run_path(str(PROJECT / "hatch_build.py"))
    package = tmp_path / "clap_cpp"
    (package / "licenses").mkdir(parents=True)
    (package / "libclap.dylib").write_bytes(b"single-architecture packaging fixture")
    (package / "licenses" / "clap-cpp.txt").write_text("Runtime notice")
    hook = module["CustomBuildHook"](
        str(tmp_path), {}, None, SimpleNamespace(name="clap-cpp"), str(tmp_path), "wheel")
    monkeypatch.setattr(module["sys"], "platform", "darwin")
    # Model a universal2 interpreter, including sysconfig's explicit host override.
    monkeypatch.setattr(module["sysconfig"], "get_platform", lambda: os.environ.get(
        "_PYTHON_HOST_PLATFORM", "macosx-11.0-universal2"))
    monkeypatch.delenv("_PYTHON_HOST_PLATFORM", raising=False)
    return hook


@pytest.mark.parametrize("explicit", [False, True])
def test_macos_universal2_tag_rejected(macos_hook, monkeypatch, explicit):
    if explicit:
        monkeypatch.setenv("_PYTHON_HOST_PLATFORM", "macosx-11.0-universal2")
    build_data = {}
    with pytest.raises(RuntimeError, match="universal2") as error:
        macos_hook.initialize("standard", build_data)
    assert "_PYTHON_HOST_PLATFORM" in str(error.value)
    assert "single-architecture" in str(error.value)
    assert "staged runtime" in str(error.value)
    assert "deployment target" in str(error.value)
    assert "tag" not in build_data


def test_macos_explicit_arm64_runtime_tag(macos_hook, monkeypatch):
    monkeypatch.setenv("_PYTHON_HOST_PLATFORM", "macosx-11.0-arm64")
    build_data = {}
    macos_hook.initialize("standard", build_data)
    assert build_data == {"pure_python": False, "tag": "py3-none-macosx_11_0_arm64"}


def test_template_matches_checked_in_cpu_metadata(tmp_path):
    project, _ = render(PROJECT, tmp_path / "cpu")
    assert (project / "pyproject.toml").read_bytes() == (PROJECT / "pyproject.toml").read_bytes()
    config = metadata(project)
    assert config["build-system"] == {"requires": ["hatchling>=1.27"], "build-backend": "hatchling.build"}
    assert "scikit-build" not in config["tool"]
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["hooks"]["custom"]["path"] == "hatch_build.py"


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm", "vulkan"])
def test_wheel_and_wrapper_only_sdist(tmp_path, backend):
    project, package = render(PROJECT, tmp_path / "project", backend, "2.3.4rc1")
    config = metadata(project)
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
    for tool in ("cmake", "ninja", "cc", "c++"):
        executable = tools / tool
        executable.write_text(f'#!/bin/sh\nprintf called > "{marker}"\nexit 91\n')
        executable.chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""))
    out = tmp_path / "dist"
    metadata_dir = project / "metadata"
    metadata_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", "from hatchling.build import prepare_metadata_for_build_wheel; "
         "prepare_metadata_for_build_wheel('metadata')"],
        cwd=project, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(metadata_dir.glob("*.dist-info/METADATA"))
    result = build(project, out, "wheel", env)
    assert result.returncode != 0 and "Missing staged external runtime" in result.stdout, result.stdout
    assert not list(out.glob("*.whl"))
    native = project / package / RUNTIME_NAME
    native.touch()
    result = build(project, out, "wheel", env)
    assert result.returncode != 0 and "Missing staged external runtime" in result.stdout, result.stdout
    native.write_bytes(b"packaging fixture only, NOT an executable runtime")
    result = build(project, out, "wheel", env)
    assert result.returncode != 0 and "Missing runtime notices" in result.stdout, result.stdout
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
        assert not any("__pycache__" in n for n in entries)
    result = build(project, out, "sdist", env)
    assert result.returncode == 0, result.stdout
    with tarfile.open(next(out.glob("*.tar.gz"))) as archive:
        entries = {n.split("/", 1)[1] for n in archive.getnames() if "/" in n}
        required = {"pyproject.toml", "pyproject.toml.j2", "README.md", "LICENSE", "hatch_build.py",
                    *(f"{package}/{name}" for name in ("__init__.py", "clap.py", "clap.pyi", "py.typed"))}
        assert required <= entries
        assert not any(n.startswith("native/") or "licenses/" in n or "companions/" in n or
                       n.endswith((".so", ".dll", ".dylib", ".cpp")) for n in entries)
        assert not {"stage_runtime.py", "prepare_backend.py", "_build_backend.py"} & entries
        archive.extractall(tmp_path / "extracted", filter="data")
    source = next((tmp_path / "extracted").iterdir())
    result = build(source, tmp_path / "from-sdist", "wheel", env)
    assert result.returncode != 0 and "Missing staged external runtime" in result.stdout, result.stdout
    shutil.copytree(project / package, source / package, dirs_exist_ok=True)
    result = build(source, tmp_path / "from-sdist", "wheel", env)
    assert result.returncode == 0, result.stdout
    if backend == "cpu":
        render(source, tmp_path / "cuda-from-cpu-sdist", "cuda")
    assert not marker.exists()


def test_editable_requires_runtime(tmp_path):
    project, package = render(PROJECT, tmp_path / "project")
    command = [sys.executable, "-c", "from hatchling.build import build_editable; build_editable('editable')"]
    result = subprocess.run(command, cwd=project, capture_output=True, text=True)
    assert result.returncode != 0 and "Missing staged external runtime" in result.stderr
    (project / package / RUNTIME_NAME).write_bytes(b"editable fixture")
    result = subprocess.run(command, cwd=project, capture_output=True, text=True)
    assert result.returncode != 0 and "Missing runtime notices" in result.stderr
