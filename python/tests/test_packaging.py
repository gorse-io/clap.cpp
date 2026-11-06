"""Package-only wheels consume an explicitly staged external C API runtime."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_project(destination):
    shutil.copytree(PROJECT, destination, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "dist", "build", "*.so", "*.dylib", "*.dll", "licenses"
    ))
    return destination


def build(project, out, kind, env=None):
    return subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", f"--{kind}", "--outdir", str(out), str(project)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )


def test_package_only_settings(tmp_path, monkeypatch):
    from scikit_build_core.settings.skbuild_read_settings import SettingsReader

    monkeypatch.chdir(tmp_path)
    config = metadata()
    assert config["build-system"]["build-backend"] == "_build_backend"
    assert config["build-system"]["backend-path"] == ["."]
    assert "cmake" not in config["tool"]["scikit-build"]
    assert "force-include" not in config["tool"]["scikit-build"]["sdist"]
    for state in ("wheel", "editable", "sdist"):
        reader = SettingsReader(config, {}, state=state)
        reader.validate_may_exit()
        settings = reader.settings
        assert settings.wheel.cmake is False
        assert settings.wheel.platlib is True
        assert settings.wheel.py_api == "py3"
        assert settings.sdist.cmake is False


@pytest.mark.parametrize("backend", ["cuda", "rocm", "vulkan"])
def test_backend_projects_wrapper_only(tmp_path, backend):
    destination = tmp_path / backend
    load("generator", PROJECT / "prepare_backend.py").prepare(backend, destination)
    config = metadata(destination)
    assert config["project"]["name"] == f"clap-cpp-{backend}"
    assert config["project"]["version"] == metadata()["project"]["version"]
    assert config["project"]["dependencies"] == []
    settings = config["tool"]["scikit-build"]
    assert settings["wheel"] == dict(packages=[f"clap_cpp_{backend}"], **{
        "py-api": "py3", "cmake": False, "platlib": True,
        "force-include": {f"clap_cpp_{backend}": f"clap_cpp_{backend}"},
    })
    assert "cmake" not in settings
    assert not (destination / "native").exists()
    assert not list(destination.rglob("CMakeLists.txt"))
    for name in ("stage_runtime.py", "_build_backend.py", "LICENSE"):
        assert (destination / name).read_bytes() == (PROJECT / name).read_bytes()
    for name in ("clap.py", "clap.pyi", "py.typed"):
        assert (destination / f"clap_cpp_{backend}" / name).read_bytes() == (PROJECT / "clap_cpp" / name).read_bytes()
    result = build(destination, tmp_path / "missing-wheel", "wheel")
    assert result.returncode != 0 and "stage_runtime.py" in result.stdout
    result = build(destination, tmp_path / "dist", "sdist")
    assert result.returncode == 0, result.stdout
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as archive:
        names = archive.getnames()
        assert any(n.endswith("/stage_runtime.py") for n in names)
        assert any(n.endswith("/_build_backend.py") for n in names)
        assert not any("/native/" in n or n.endswith((".so", ".dll", ".dylib")) for n in names)
        archive.extractall(tmp_path / "extracted", filter="data")
    source = next((tmp_path / "extracted").iterdir())
    native = "clap.dll" if sys.platform == "win32" else ("libclap.dylib" if sys.platform == "darwin" else "libclap.so")
    runtime = tmp_path / "runtime" / "clap_cpp"
    runtime.mkdir(parents=True)
    (runtime / native).write_bytes(b"backend packaging fixture, not executable")
    load("backend_stage", source / "stage_runtime.py").stage(source, runtime.parent, backend)
    result = build(source, tmp_path / "staged-wheel", "wheel")
    assert result.returncode == 0, result.stdout
    wheel = next((tmp_path / "staged-wheel").glob("*.whl"))
    assert "-py3-none-" in wheel.name and not wheel.name.endswith("-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read(f"clap_cpp_{backend}/{native}") == (runtime / native).read_bytes()
        assert f"clap_cpp_{backend}/licenses/clap-cpp.txt" in archive.namelist()


@pytest.mark.parametrize("backend", [None, "cuda", "rocm", "vulkan"])
def test_stage_runtime(tmp_path, backend):
    stage = load("stage", PROJECT / "stage_runtime.py")
    project = tmp_path / "project"
    if backend:
        load("generator", PROJECT / "prepare_backend.py").prepare(backend, project)
    else:
        copy_project(project)
    runtime = tmp_path / "runtime"
    (runtime / "clap_cpp" / "licenses").mkdir(parents=True)
    (runtime / "clap_cpp" / "libclap.so").write_bytes(b"unit-test runtime, not executable")
    (runtime / "clap_cpp" / "licenses" / "icu.txt").write_text("ICU notice")
    (runtime / "clap_cpp" / "licenses" / "other.txt").write_text("companion notice")
    (runtime / "clap_cpp" / "companions").mkdir()
    (runtime / "clap_cpp" / "companions" / "libdependency.so.1").write_bytes(b"test companion")
    stage.stage(project, runtime, backend=backend)
    package = project / (f"clap_cpp_{backend}" if backend else "clap_cpp")
    assert (package / "libclap.so").read_bytes() == (runtime / "clap_cpp" / "libclap.so").read_bytes()
    assert (package / "licenses" / "icu.txt").read_text() == "ICU notice"
    assert (package / "licenses" / "other.txt").read_text() == "companion notice"
    assert (package / "companions" / "libdependency.so.1").exists()
    assert (package / "licenses" / "clap-cpp.txt").read_bytes() == (PROJECT / "LICENSE").read_bytes()
    assert (package / "clap.py").read_bytes() == (PROJECT / "clap_cpp" / "clap.py").read_bytes()


@pytest.mark.parametrize("sibling", ["clap_cpp.libs", "licenses"])
def test_stage_rejects_sibling_directories_without_relocation(tmp_path, sibling):
    project = copy_project(tmp_path / "project")
    runtime = tmp_path / "runtime"
    (runtime / "clap_cpp").mkdir(parents=True)
    (runtime / "clap_cpp" / RUNTIME_NAME).write_bytes(b"packaging fixture")
    (runtime / sibling).mkdir()
    (runtime / sibling / "companion.so").write_bytes(b"sibling payload")
    with pytest.raises(ValueError, match="sibling"):
        load("stage", PROJECT / "stage_runtime.py").stage(project, runtime)
    assert not (project / "clap_cpp" / RUNTIME_NAME).exists()
    assert not (project / "clap_cpp" / sibling).exists()
    assert not (project / sibling).exists()


def test_missing_runtime_and_wrapper_collision(tmp_path):
    stage = load("stage", PROJECT / "stage_runtime.py")
    project = copy_project(tmp_path / "project")
    with pytest.raises((ValueError, FileNotFoundError), match="runtime"):
        stage.stage(project, tmp_path / "absent")
    runtime = tmp_path / "runtime"
    (runtime / "clap_cpp").mkdir(parents=True)
    with pytest.raises((ValueError, FileNotFoundError), match="runtime"):
        stage.stage(project, runtime)
    (runtime / "clap_cpp" / "libclap.so").write_bytes(b"test")
    (runtime / "clap_cpp" / "clap.py").write_text("overwrite")
    with pytest.raises(ValueError, match="wrapper"):
        stage.stage(project, runtime)
    assert not (project / "clap_cpp" / "libclap.so").exists()


def test_missing_and_staged_wheel_sdist_contract(tmp_path):
    project = copy_project(tmp_path / "project")
    out = tmp_path / "dist"
    # Any attempt to execute CMake/Ninja fails and leaves visible evidence.
    tools = tmp_path / "tools"
    tools.mkdir()
    marker = tmp_path / "compiler-called"
    for tool in ("cmake", "ninja", "cc", "c++"):
        executable = tools / tool
        executable.write_text(f'#!/bin/sh\nprintf called > "{marker}"\nexit 91\n')
        executable.chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""))
    result = subprocess.run(
        [sys.executable, "-c", "import _build_backend as b; "
         "assert b.get_requires_for_build_wheel() == []; "
         "assert b.get_requires_for_build_sdist() == []; "
         "b.prepare_metadata_for_build_wheel('metadata')"],
        cwd=project, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    assert list((project / "metadata").glob("*.dist-info/METADATA"))
    result = build(project, out, "wheel", env)
    assert result.returncode != 0
    assert "stage_runtime.py" in result.stdout
    assert not list(out.glob("*.whl"))
    runtime = tmp_path / "runtime" / "clap_cpp"
    runtime.mkdir(parents=True)
    (runtime / RUNTIME_NAME).write_bytes(b"unit packaging payload only, not inference")
    load("stage", PROJECT / "stage_runtime.py").stage(project, runtime.parent)
    result = build(project, out, "wheel", env)
    assert result.returncode == 0, result.stdout
    wheel = next(out.glob("*.whl"))
    assert "-py3-none-" in wheel.name and not wheel.name.endswith("-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        wheel_metadata = archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/WHEEL"))).decode()
        assert "Root-Is-Purelib: false" in wheel_metadata
        assert archive.read(f"clap_cpp/{RUNTIME_NAME}") == (runtime / RUNTIME_NAME).read_bytes()
        assert "clap_cpp/clap.pyi" in archive.namelist()
        assert "clap_cpp/py.typed" in archive.namelist()
        assert "clap_cpp/licenses/clap-cpp.txt" in archive.namelist()
    result = build(project, out, "sdist", env)
    assert result.returncode == 0, result.stdout
    extracted = tmp_path / "extracted"
    with tarfile.open(next(out.glob("*.tar.gz"))) as archive:
        names = archive.getnames()
        assert len(names) == len(set(names))
        assert not any("/native/" in n or n.endswith((".so", ".dylib", ".dll", ".cpp")) for n in names)
        for name in ("stage_runtime.py", "prepare_backend.py", "_build_backend.py", "LICENSE", "clap_cpp/clap.pyi", "clap_cpp/py.typed"):
            assert any(n.endswith("/" + name) for n in names)
        archive.extractall(extracted, filter="data")
    source = next(extracted.iterdir())
    result = build(source, tmp_path / "from-sdist", "wheel", env)
    assert result.returncode != 0 and "stage_runtime.py" in result.stdout
    load("sdist_stage", source / "stage_runtime.py").stage(source, runtime.parent)
    result = build(source, tmp_path / "from-sdist", "wheel", env)
    assert result.returncode == 0, result.stdout
    load("sdist_backend", source / "prepare_backend.py").prepare("cuda", tmp_path / "cuda")
    assert not marker.exists()
