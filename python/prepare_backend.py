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
    # Preserve upstream source directories named build/dist too: the pinned
    # manifest, not generic directory names, defines the source snapshot.
    ignore = shutil.ignore_patterns(".git", "__pycache__")
    for original, target in entries:
        # Inside a base sdist the manifest's ../ paths have already been moved.
        origin = source / target if (source / "native").is_dir() else source / original
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, output, ignore=ignore)
        else:
            shutil.copy2(origin, output)
    shutil.copy2(source / "README.md", destination / "README.md")
    (destination / package).mkdir()
    # All backends use the identical ctypes loader, relative to their own package.
    for name in ("clap.py", "clap.pyi", "py.typed"):
        shutil.copy2(source / "clap_cpp" / name, destination / package / name)
    (destination / package / "__init__.py").write_text(
        f'"""Native {backend} runtime; import clap_cpp for the public API."""\n', encoding="utf-8"
    )
    # Inherit all portable CPU/dependency switches from the canonical project;
    # only the selected GGML backend and runtime installation namespace differ.
    defines = metadata.split("[tool.scikit-build.cmake.define]", 1)[1].split("\n[", 1)[0]
    defines = defines.replace('"clap_cpp"', f'"{package}"')
    defines = defines.replace(f'GGML_{cmake_backend} = "OFF"', f'GGML_{cmake_backend} = "ON"')
    (destination / "pyproject.toml").write_text(
        f'''[build-system]
requires = ["scikit-build-core>=1.0"]
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
wheel.py-api = "py3"
install.components = ["runtime"]
cmake.source-dir = "native"
cmake.version = ">=3.20"
# Preserve the exact pinned source snapshot, without upstream .gitignore rules.
sdist.inclusion-mode = "manual"
sdist.exclude = ["**/.git", "**/__pycache__/**", "dist/**", "build/**"]

[tool.scikit-build.cmake.define]{defines}
''', encoding="utf-8"
    )
    print(f"Prepared clap-cpp-{backend} {version} in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("cuda", "rocm", "vulkan"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.backend, args.destination.resolve())
