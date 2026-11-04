"""The Python distributions share the repository's native build entrypoint."""

import importlib.util
from pathlib import Path
import shutil

import pytest

try:
    import tomllib
except ImportError:
    tomllib = pytest.importorskip("tomli")


PROJECT = Path(__file__).resolve().parents[1]


def metadata(path=PROJECT):
    return tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))


def test_root_source_manifest():
    config = metadata()["tool"]["scikit-build"]
    manifest = config["sdist"]["force-include"]
    assert manifest["../CMakeLists.txt"] == "native/CMakeLists.txt"
    for directory in ("include", "src", "cmake", "ggml"):
        assert manifest[f"../{directory}"] == f"native/{directory}"
    assert not (PROJECT / "CMakeLists.txt").exists()
    assert config["install"]["components"] == ["runtime"]
    assert config["wheel"]["py-api"] == "py3"
    assert not config.get("build", {}).get("targets")
    defines = config["cmake"]["define"]
    for key in ("CLAP_BUILD_CLI", "BUILD_TESTING", "GGML_NATIVE", "GGML_OPENMP",
                "GGML_METAL", "GGML_CUDA", "GGML_HIP", "GGML_VULKAN", "GGML_AVX2"):
        assert defines[key] == "OFF"
    for key in ("BUILD_SHARED_LIBS", "CLAP_STATIC_DEPENDENCIES"):
        assert defines[key] == "ON"
    assert defines["CMAKE_INSTALL_LIBDIR"] == defines["CMAKE_INSTALL_BINDIR"] == "clap_cpp"
    assert not any(key.startswith("CLAP_PYTHON") for key in defines)


def test_supported_checkout_and_sdist_source_dirs(tmp_path, monkeypatch):
    reader = pytest.importorskip("scikit_build_core.settings.skbuild_read_settings")
    monkeypatch.chdir(tmp_path)
    for from_sdist, expected in ((False, ".."), (True, "native")):
        if from_sdist:
            (tmp_path / "PKG-INFO").write_text("Metadata-Version: 2.1\n")
        for state in ("wheel", "editable", "sdist"):
            settings = reader.SettingsReader(metadata(), {}, state=state).settings
            assert str(settings.cmake.source_dir) == expected


@pytest.mark.parametrize("from_sdist", [False, True])
@pytest.mark.parametrize("backend,flag", [("cuda", "CUDA"), ("rocm", "HIP"), ("vulkan", "VULKAN")])
def test_backend_projects_share_root_manifest(tmp_path, from_sdist, backend, flag):
    # Tiny source snapshot keeps these packaging regressions independent of
    # submodules, compilers, native runtimes and GPU toolkits.
    source = tmp_path / "checkout" / "python"
    source.mkdir(parents=True)
    shutil.copy2(PROJECT / "prepare_backend.py", source)
    shutil.copy2(PROJECT / "pyproject.toml", source)
    (source / "README.md").write_text("test package\n")
    manifest = metadata()["tool"]["scikit-build"]["sdist"]["force-include"]
    expected = {}
    for original, target in manifest.items():
        path = source / (target if from_sdist else original)
        if Path(target).suffix or Path(target).name == "LICENSE":
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            path /= "fixture.txt"
        path.write_text(f"pinned {target}\n")
        expected[str(Path(target) / "fixture.txt") if path.name == "fixture.txt" else target] = path.read_bytes()
    (source / "clap_cpp").mkdir()
    for name in ("clap.py", "clap.pyi", "py.typed"):
        (source / "clap_cpp" / name).write_text(f"# {name}\n")
    spec = importlib.util.spec_from_file_location("backend_generator", source / "prepare_backend.py")
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    destination = tmp_path / "backend"
    generator.prepare(backend, destination)
    config = metadata(destination)
    assert config["project"]["name"] == f"clap-cpp-{backend}"
    assert config["project"]["version"] == metadata()["project"]["version"]
    assert config["project"]["dependencies"] == []
    build = config["tool"]["scikit-build"]
    assert build["cmake"]["source-dir"] == "native"
    assert build["install"]["components"] == ["runtime"]
    assert build["wheel"]["py-api"] == "py3"
    assert build["sdist"]["inclusion-mode"] == "manual"
    assert "force-include" not in build["sdist"]
    defines = build["cmake"]["define"]
    expected_defines = metadata()["tool"]["scikit-build"]["cmake"]["define"].copy()
    expected_defines.update({f"GGML_{flag}": "ON", "CMAKE_INSTALL_LIBDIR": f"clap_cpp_{backend}",
                             "CMAKE_INSTALL_BINDIR": f"clap_cpp_{backend}"})
    assert defines == expected_defines
    assert not (destination / "CMakeLists.txt").exists()
    for target, content in expected.items():
        assert (destination / target).read_bytes() == content
    for name in ("clap.py", "clap.pyi", "py.typed"):
        assert (destination / f"clap_cpp_{backend}" / name).read_bytes() == (source / "clap_cpp" / name).read_bytes()
