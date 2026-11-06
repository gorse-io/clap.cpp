"""Thin scikit-build-core guard: a wheel must have a staged native runtime.

Metadata and source distributions need no runtime. Wheel/editable construction
checks the package-only contract before delegating; no native build fallback.
"""

from pathlib import Path
import sys

from scikit_build_core import build as _build
from scikit_build_core.settings.skbuild_read_settings import SettingsReader

from stage_runtime import RUNTIME_NAMES


# Reading the project here keeps CPU and generated backend projects identical.
try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _guard(config_settings, state):
    with Path("pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    reader = SettingsReader(metadata, config_settings or {}, state=state)
    reader.validate_may_exit()
    settings = reader.settings
    if settings.wheel.cmake or not settings.wheel.platlib or settings.wheel.py_api != "py3":
        raise RuntimeError("Python wheels require wheel.cmake=false, wheel.platlib=true, wheel.py-api=py3")
    packages = settings.wheel.packages
    expected = metadata["project"]["name"].replace("-", "_")
    if packages != [expected]:
        raise RuntimeError(f"Expected wheel.packages=[{expected!r}] for this runtime project")
    package = Path(expected)
    native = "clap.dll" if sys.platform == "win32" else (
        "libclap.dylib" if sys.platform == "darwin" else "libclap.so"
    )
    if native not in RUNTIME_NAMES or not (package / native).is_file() or (package / native).stat().st_size == 0:
        backend = expected.removeprefix("clap_cpp_") if expected != "clap_cpp" else None
        option = f" --backend {backend}" if backend else ""
        raise RuntimeError(
            f"Missing staged external runtime: {package / native}. "
            f"Run: python stage_runtime.py . /path/to/libraries/TARGET{option}. "
            "Python packaging does not compile native libraries."
        )
    if not (package / "licenses" / "clap-cpp.txt").is_file():
        raise RuntimeError("Missing runtime notices; run stage_runtime.py before building a wheel")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _guard(config_settings, "wheel")
    return _build.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _guard(config_settings, "editable")
    return _build.build_editable(wheel_directory, config_settings, metadata_directory)


build_sdist = _build.build_sdist
get_requires_for_build_sdist = _build.get_requires_for_build_sdist
get_requires_for_build_wheel = _build.get_requires_for_build_wheel
get_requires_for_build_editable = _build.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _build.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _build.prepare_metadata_for_build_editable
