# Copyright 2026 clap.cpp Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loader tests need no driver, toolkit or model; CDLL is the native boundary."""
import ctypes
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


WRAPPER = Path(__file__).resolve().parents[1] / "clap_cpp" / "clap.py"
LIBRARIES = ("libnvrtc.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcudart.so.12")
PACKAGES = ("cuda_nvrtc", "cublas", "cublas", "cuda_runtime")


@pytest.fixture(params=["namespace", "regular"])
def dependencies(tmp_path, monkeypatch, request):
    # Separate roots exercise a PEP 420 namespace (nvidia.__file__ is None).
    roots = {}
    for package in ("cuda_nvrtc", "cublas", "cuda_runtime"):
        root = tmp_path / package
        directory = root / "nvidia" / package / "lib"
        directory.mkdir(parents=True)
        if request.param == "regular":
            (directory.parent / "__init__.py").touch()
        roots[package] = directory
        monkeypatch.syspath_prepend(str(root))
    for name in list(sys.modules):
        if name == "nvidia" or name.startswith("nvidia."):
            monkeypatch.delitem(sys.modules, name)
    for package, filename in zip(PACKAGES, LIBRARIES):
        (roots[package] / filename).touch()
    yield roots
    for name in list(sys.modules):
        if name == "nvidia" or name.startswith("nvidia."):
            monkeypatch.delitem(sys.modules, name)


def load_wrapper(monkeypatch, backend="cuda", platform="linux", fail=None):
    calls = []
    handles = []

    def cdll(path, **kwargs):
        calls.append((Path(path), kwargs))
        if fail and Path(path).name == fail:
            raise OSError("native dependency rejected")
        handle = SimpleNamespace(**{
            name: SimpleNamespace() for name in (
                "clap_default_options", "clap_last_error", "clap_create", "clap_destroy",
                "clap_available_devices", "clap_free_devices", "clap_free_audio_frames",
                "clap_encode_text", "clap_encode_audio", "clap_encode_audio_file",
                "clap_get_audio_frames", "clap_audio_sample_rate", "clap_projection_dimension",
            )
        })
        handles.append(handle)
        return handle

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    monkeypatch.setattr(sys, "platform", platform)
    namespace = "clap_cpp" if backend == "cpu" else f"clap_cpp_{backend}"
    spec = importlib.util.spec_from_file_location(f"{namespace}.clap", WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        error.calls = calls
        raise
    return module, calls, handles


def test_cuda_preloads_installed_namespace_dependencies_in_order(dependencies, monkeypatch):
    module, calls, handles = load_wrapper(monkeypatch)
    expected = [dependencies[package] / filename for package, filename in zip(PACKAGES, LIBRARIES)]
    expected.append(WRAPPER.parent / "libclap.so")
    assert [path for path, _ in calls] == expected
    assert all(kwargs == {"mode": ctypes.RTLD_GLOBAL} for _, kwargs in calls[:4])
    assert module._cuda_dependencies == handles[:4]  # Keep foreign handles alive.
    assert module._lib is handles[4]


@pytest.mark.parametrize("backend,platform", [
    ("cpu", "linux"), ("rocm", "linux"), ("vulkan", "linux"),
    ("cuda", "darwin"), ("cuda", "win32"),
])
def test_other_backends_do_not_discover_nvidia(monkeypatch, backend, platform):
    import builtins
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("nvidia") or name.startswith("importlib"):
            raise AssertionError(f"Premature NVIDIA dependency discovery: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    _, calls, _ = load_wrapper(monkeypatch, backend, platform)
    assert len(calls) == 1
    assert calls[0][0].parent == WRAPPER.parent


@pytest.mark.parametrize("missing", LIBRARIES)
def test_missing_cuda_library_fails_without_system_fallback(dependencies, monkeypatch, missing):
    directory = dependencies[PACKAGES[LIBRARIES.index(missing)]]
    (directory / missing).unlink()
    with pytest.raises(ImportError, match=missing) as caught:
        load_wrapper(monkeypatch)
    assert all(path.is_absolute() and path.name != "libclap.so" for path, _ in caught.value.calls)
    assert "nvidia-cuda-runtime-cu12" in str(caught.value)
    assert "nvidia-cublas-cu12" in str(caught.value)
    assert "nvidia-cuda-nvrtc-cu12==12.9.86" in str(caught.value)


@pytest.mark.parametrize("package", ["cuda_nvrtc", "cuda_runtime"])
def test_missing_cuda_package_has_actionable_error(dependencies, monkeypatch, package):
    # Hide one entire namespace package, not just its shared library.
    dependencies[package].parent.rename(dependencies[package].parent.with_name("hidden"))
    with pytest.raises(ImportError, match=f"nvidia-{package.replace('_', '-')}-cu12"):
        load_wrapper(monkeypatch)


@pytest.mark.parametrize("unloadable", LIBRARIES)
def test_unloadable_cuda_dependency_preserves_cause(dependencies, monkeypatch, unloadable):
    with pytest.raises(ImportError, match=unloadable) as caught:
        load_wrapper(monkeypatch, fail=unloadable)
    assert isinstance(caught.value.__cause__, OSError)
    assert [p.name for p, _ in caught.value.calls] == list(LIBRARIES[:LIBRARIES.index(unloadable) + 1])
