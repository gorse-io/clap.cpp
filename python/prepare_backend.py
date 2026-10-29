"""Create a standalone GPU wheel project from a checkout or the base sdist."""

import argparse
from pathlib import Path
import re
import shutil


def prepare(backend, destination):
    source = Path(__file__).resolve().parent
    metadata = (source / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE)
    if match is None:
        raise ValueError("Missing package version")
    version = match.group(1)
    package = f"clap_cpp_{backend}"
    cmake_backend = "HIP" if backend == "rocm" else backend.upper()
    # Reuse the base sdist's exact native-source manifest, including licenses.
    manifest = metadata.split("[tool.scikit-build.sdist.force-include]", 1)[1].split("\n[", 1)[0]
    entries = re.findall(r'^"([^"]+)" = "([^"]+)"$', manifest, re.MULTILINE)
    if not entries:
        raise ValueError("Missing native source manifest")
    destination.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns(".git", "__pycache__", "build", "dist")
    for original, target in entries:
        # Inside a base sdist the manifest's ../ paths have already been moved.
        origin = source / target if (source / "native").is_dir() else source / original
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, output, ignore=ignore)
        else:
            shutil.copy2(origin, output)
    for name in ("CMakeLists.txt", "bindings.cpp", "README.md"):
        shutil.copy2(source / name, destination / name)
    (destination / package).mkdir()
    (destination / package / "__init__.py").write_text(
        f'"""Native {backend} runtime; import clap_cpp for the public API."""\n', encoding="utf-8"
    )
    (destination / "pyproject.toml").write_text(
        f'''[build-system]
requires = ["scikit-build-core>=1.0", "pybind11>=2.13"]
build-backend = "scikit_build_core.build"

[project]
name = "clap-cpp-{backend}"
version = "{version}"
description = "{backend.upper()} backend for clap-cpp"
readme = "README.md"
requires-python = ">=3.9"
license = "MIT"
dependencies = []

[tool.scikit-build]
minimum-version = "build-system.requires"
wheel.packages = ["{package}"]
install.components = ["python"]
build.targets = ["_clap_cpp"]
sdist.exclude = ["**/.git", "**/__pycache__/**", "dist/**", "build/**"]

[tool.scikit-build.cmake.define]
CLAP_PYTHON_BACKEND = "{cmake_backend}"
''', encoding="utf-8"
    )
    print(f"Prepared clap-cpp-{backend} {version} in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("cuda", "rocm", "vulkan"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.backend, args.destination.resolve())
