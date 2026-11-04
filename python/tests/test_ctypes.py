"""Migration and ABI regression checks; native integration lives in test_bindings."""
import ctypes
import importlib.util
from pathlib import Path
import shutil

import pytest


SOURCE = Path(__file__).parents[1] / "clap_cpp" / "clap.py"


def test_wrapper_is_python_source():
    assert SOURCE.is_file(), "the pybind bridge must be replaced by Python ctypes"


def test_missing_native_is_package_local(tmp_path):
    # Loading by filename also exercises the generated backend-package contract.
    target = tmp_path / "clap.py"
    target.write_bytes(SOURCE.read_bytes())
    spec = importlib.util.spec_from_file_location("isolated_clap", target)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError, match="package-local"):
        spec.loader.exec_module(module)


def test_ctypes_abi_and_lock():
    from clap_cpp import clap as wrapper
    assert isinstance(wrapper._lib, ctypes.CDLL)
    assert wrapper._Device._fields_[-1] == ("gpu", ctypes.c_bool)
    assert wrapper._lib.clap_default_options.restype is wrapper._Options
    assert wrapper._lib.clap_last_error.restype is ctypes.c_char_p
    assert wrapper._lib.clap_free.restype is None
    for name in ("clap_create", "clap_destroy", "clap_available_devices",
                 "clap_free_devices", "clap_tokenize", "clap_encode_text",
                 "clap_encode_tokens", "clap_encode_audio", "clap_encode_audio_file",
                 "clap_get_audio_frames", "clap_free_audio_frames",
                 "clap_audio_sample_rate", "clap_projection_dimension"):
        assert getattr(wrapper._lib, name).argtypes is not None
    for name in ("clap_audio_sample_rate", "clap_projection_dimension"):
        query = getattr(wrapper._lib, name)
        assert query.argtypes == [ctypes.c_void_p]
        assert query.restype is ctypes.c_int
        assert query(None) == 0
    # Partial initialization and repeated finalization must never destroy garbage.
    instance = wrapper.CLAP.__new__(wrapper.CLAP)
    instance.__del__()
    instance.close()


def test_status_mapping_copies_diagnostic(monkeypatch):
    from clap_cpp import clap as wrapper
    monkeypatch.setattr(wrapper._lib, "clap_last_error", lambda: b"specific diagnostic")
    for status, exception in ((1, ValueError), (2, MemoryError), (3, RuntimeError), (99, RuntimeError)):
        with pytest.raises(exception, match="specific diagnostic"):
            wrapper._check(status)


def test_native_device_struct_layout():
    from clap_cpp import clap as wrapper
    devices = ctypes.POINTER(wrapper._Device)()
    count = ctypes.c_size_t()
    try:
        wrapper._check(wrapper._lib.clap_available_devices(
            ctypes.byref(devices), ctypes.byref(count)))
        assert count.value >= 1
        for index in range(count.value):
            assert devices[index].name
            assert isinstance(devices[index].description, bytes)
            assert type(devices[index].gpu) is bool
    finally:
        wrapper._lib.clap_free_devices(devices, count)


def test_backend_copy_is_standalone(tmp_path):
    import clap_cpp.clap as wrapper
    target = tmp_path / "clap.py"
    target.write_bytes(SOURCE.read_bytes())
    native = Path(wrapper._lib._name)
    # Copy rather than symlink: Windows test users may lack symlink privilege.
    shutil.copy2(native, tmp_path / native.name)
    spec = importlib.util.spec_from_file_location("backend_clap", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CLAP.__module__ == "backend_clap"
