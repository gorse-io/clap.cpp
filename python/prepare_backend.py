"""Create a wrapper-only GPU wheel project from a checkout or base sdist.

Stage its externally compiled runtime later using stage_runtime.py --backend.
"""

import argparse
from pathlib import Path
import re
import shutil


BACKENDS = ("cuda", "rocm", "vulkan")


def prepare(backend, destination):
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    destination = Path(destination)
    source = Path(__file__).resolve().parent
    metadata = (source / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE)
    if match is None:
        raise ValueError("Missing package version")
    version = match.group(1)
    package = f"clap_cpp_{backend}"
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("README.md", "LICENSE", "stage_runtime.py", "_build_backend.py"):
        shutil.copy2(source / name, destination / name)
    (destination / package).mkdir()
    for name in ("clap.py", "clap.pyi", "py.typed"):
        shutil.copy2(source / "clap_cpp" / name, destination / package / name)
    (destination / package / "__init__.py").write_text(
        f'"""Native {backend} runtime; import clap_cpp for the public API."""\n', encoding="utf-8"
    )
    (destination / "pyproject.toml").write_text(
        f'''[build-system]
requires = ["scikit-build-core>=1.1.2"]
build-backend = "_build_backend"
backend-path = ["."]

[project]
name = "clap-cpp-{backend}"
version = "{version}"
description = "{backend.upper()} backend for clap-cpp"
readme = "README.md"
requires-python = ">=3.9"
license = "MIT"
license-files = ["LICENSE"]
dependencies = []

[tool.scikit-build]
minimum-version = "build-system.requires"
wheel.packages = ["{package}"]
wheel.force-include = {{ {package} = "{package}" }}
wheel.cmake = false
wheel.platlib = true
wheel.py-api = "py3"
sdist.cmake = false
sdist.inclusion-mode = "explicit"
sdist.include = ["pyproject.toml", "README.md", "LICENSE", "_build_backend.py", "stage_runtime.py", "{package}/*.py", "{package}/*.pyi", "{package}/py.typed"]
sdist.exclude = ["**/__pycache__/**"]
''', encoding="utf-8"
    )
    print(f"Prepared clap-cpp-{backend} {version} in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=BACKENDS)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.backend, args.destination.resolve())
