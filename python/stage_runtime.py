"""Stage an external runtime: stage_runtime.py PROJECT_DIR RUNTIME_DIR [--backend NAME].

RUNTIME_DIR contains only clap_cpp/, the common native install namespace.
Its contents go into the target Python namespace without changing package-local
relative paths. Sibling payloads such as clap_cpp.libs are explicitly rejected:
relocating them would break loader paths like $ORIGIN/../clap_cpp.libs. No
compilation or native source snapshot is performed by Python packaging.
"""

import argparse
from pathlib import Path
import shutil


BACKENDS = ("cuda", "rocm", "vulkan")
RUNTIME_NAMES = ("libclap.so", "libclap.dylib", "clap.dll")
WRAPPERS = ("__init__.py", "clap.py", "clap.pyi", "py.typed")


def stage(project_dir, runtime_dir, backend=None):
    """Copy binaries/notices into an existing wrapper project; reject missing input."""
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    project = Path(project_dir).resolve()
    runtime = Path(runtime_dir).resolve()
    package = project / (f"clap_cpp_{backend}" if backend else "clap_cpp")
    if not (project / "pyproject.toml").is_file() or not all(
        (package / name).is_file() for name in WRAPPERS
    ):
        raise ValueError(f"Not a matching wrapper project: {package}")
    source = runtime / "clap_cpp"
    if not source.is_dir() or not any((source / name).is_file() for name in RUNTIME_NAMES):
        raise FileNotFoundError(
            f"Missing external runtime in {source}: expected " + ", ".join(RUNTIME_NAMES)
        )
    siblings = sorted(path.name for path in runtime.iterdir() if path.name != "clap_cpp")
    if siblings:
        raise ValueError(
            f"Unsupported runtime sibling payloads: {', '.join(siblings)}. "
            "Only package-local native payloads beneath clap_cpp/ are accepted; "
            "sibling library directories require separate loader-aware wheel packaging."
        )
    if runtime == project or runtime in package.parents or package in runtime.parents:
        raise ValueError("External runtime and target package must not overlap")
    license_file = project / "LICENSE"
    if not license_file.is_file():
        raise FileNotFoundError(f"Missing project LICENSE: {license_file}")
    # Validate the complete copy before touching the destination. Never let a
    # runtime artifact replace the wrapper, even accidentally.
    files = {}
    for origin in sorted(source.rglob("*")):
        if not origin.is_file():
            continue
        try:
            origin.resolve().relative_to(runtime)
        except ValueError:
            raise ValueError(f"Runtime symlink escapes runtime directory: {origin}") from None
        relative = origin.relative_to(source)
        if relative.parts[0] in WRAPPERS or origin.suffix in (".py", ".pyi", ".pyc"):
            raise ValueError(f"External runtime must not contain Python wrapper sources: {origin}")
        if relative in files:
            raise ValueError(f"Runtime paths collide after namespace mapping: {relative}")
        target = package / relative
        if not target.resolve().is_relative_to(package):
            raise ValueError(f"Runtime target escapes package: {target}")
        files[relative] = origin
    # Restaging replaces the old native payload, so switching runtimes cannot
    # leave stale dependencies/notices in the next wheel. Keep wrapper files.
    for existing in package.iterdir():
        if existing.name in WRAPPERS:
            continue
        if existing.is_dir() and not existing.is_symlink():
            shutil.rmtree(existing)
        else:
            existing.unlink()
    for relative, origin in files.items():
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
    notices = package / "licenses"
    notices.mkdir(exist_ok=True)
    shutil.copy2(license_file, notices / "clap-cpp.txt")
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("runtime_dir", type=Path)
    parser.add_argument("--backend", choices=BACKENDS)
    args = parser.parse_args()
    try:
        package = stage(args.project_dir, args.runtime_dir, args.backend)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Staged external runtime in {package}")


if __name__ == "__main__":
    main()
