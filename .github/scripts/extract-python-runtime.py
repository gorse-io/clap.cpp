#!/usr/bin/env python3
"""Stage CPU wheel runtimes for npm without rewriting binaries or repair paths.

Requires packaging. Default: exactly one wheel per supported target; --target
selects a single CI target. Known GPU distributions are ignored, other names
are errors. Existing target directories are never merged or overwritten.

Native files and dependency notices retain their wheel-relative locations.
Python code, typing files, and dist-info metadata are not runtime inputs.
Windows companions must be adjacent to clap.dll: Python/delvewheel DLL search
or preload code cannot safely be transplanted into a Node package. Such layouts
fail explicitly instead of silently dropping libraries or flattening paths.
This validates archive structure, not native machine code or dynamic linkage.
"""
import argparse
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

from packaging.utils import parse_wheel_filename
from packaging.version import Version

LIBRARIES = {
    "linux-x64": "libclap.so",
    "linux-arm64": "libclap.so",
    "darwin-arm64": "libclap.dylib",
    "win32-x64": "clap.dll",
}
DEPENDENCIES = ("ggml", "miniaudio", "icu", "json")
GPU_DISTRIBUTIONS = {"clap-cpp-cuda", "clap-cpp-rocm", "clap-cpp-vulkan"}
NATIVE = re.compile(r"(?:\.so(?:\.[^/]+)*|\.dylib|\.dll|\.pyd|\.metallib)$", re.I)
DEVICES = re.compile(r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", re.I)


def wheel_target(tags):
    """All expanded wheel tags must describe the same supported py3-none target."""
    targets = set()
    for tag in tags:
        if tag.interpreter != "py3" or tag.abi != "none":
            raise ValueError(f"Unsupported Python/ABI tags: {tag}")
        platform = tag.platform
        linux = re.fullmatch(r"(?:linux|manylinux(?:1|2010|2014|_[0-9]+_[0-9]+))_(x86_64|aarch64)", platform)
        if linux:
            targets.add("linux-x64" if linux[1] == "x86_64" else "linux-arm64")
        elif re.fullmatch(r"macosx_[0-9]+_[0-9]+_arm64", platform):
            targets.add("darwin-arm64")
        elif platform == "win_amd64":
            targets.add("win32-x64")
        else:
            raise ValueError(f"Unsupported wheel platform: {platform}")
    if len(targets) != 1:
        raise ValueError(f"Unsupported mixed wheel architectures: {sorted(targets)}")
    return targets.pop()


def safe_member(info):
    """Reject ambiguous POSIX/Windows paths before filtering archive entries."""
    name = info.filename
    path = name[:-1] if info.is_dir() else name
    parts = path.split("/")
    if (not path or name != info.orig_filename or "\\" in name
            or any(ord(char) < 32 for char in name)
            or any(not part or part in {".", ".."} or part.endswith((".", " "))
                   or any(char in part for char in ':<>"|?*') or DEVICES.match(part)
                   for part in parts)):
        raise ValueError(f"Unsafe archive path: {name!r}")
    kind = stat.S_IFMT(info.external_attr >> 16)
    if kind == stat.S_IFLNK:
        raise ValueError(f"Archive symlink is forbidden: {name}")
    if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"Unsafe special archive entry: {name}")
    return path


def runtime_members(archive, target, version):
    """Return destination -> ZipInfo, after validating the complete archive."""
    members = {}
    folded = set()
    for info in archive.infolist():
        name = safe_member(info)
        if name.casefold() in folded:
            raise ValueError(f"Duplicate/case-colliding archive path: {name}")
        folded.add(name.casefold())
        members[name] = info
    for name in members:
        for parent in PurePosixPath(name).parents:
            if str(parent) in members and not members[str(parent)].is_dir():
                raise ValueError(f"Unsafe archive file/directory collision: {name}")
    core = f"clap_cpp/{LIBRARIES[target]}"
    if core not in members or members[core].is_dir() or not members[core].file_size:
        raise ValueError(f"Missing or empty expected core library: {core}")
    for dependency in DEPENDENCIES:
        name = f"clap_cpp/licenses/{dependency}.txt"
        if name not in members or members[name].is_dir() or not members[name].file_size:
            raise ValueError(f"Missing or empty native dependency license: {name}")
    selected = {}
    for name, info in members.items():
        if info.is_dir():
            continue
        path = PurePosixPath(name)
        if path.suffix.lower() in {".py", ".pyi", ".pyc", ".pyo"} or path.name == "py.typed":
            continue
        in_runtime = path.parts[0] in {"clap_cpp", "clap_cpp.libs", "clap_cpp.dylibs"}
        native = bool(NATIVE.search(name))
        if native and not in_runtime:
            raise ValueError(f"Cannot preserve native companion in unsupported location: {name}")
        if not in_runtime:
            continue
        if native:
            if path.suffix.lower() == ".pyd" or ".cpython-" in name:
                raise ValueError(f"Unexpected Python extension native companion: {name}")
            if target == "win32-x64" and path.parent != PurePosixPath("clap_cpp"):
                raise ValueError(f"Windows companion requires Python DLL search/preload logic: {name}; "
                                 "only DLLs adjacent to clap.dll can be safely reused")
            if (target.startswith("linux-") and not re.search(r"\.so(?:\.|$)", name)
                    or target == "darwin-arm64" and path.suffix not in {".dylib", ".metallib"}
                    or target == "win32-x64" and path.suffix.lower() != ".dll"):
                raise ValueError(f"Unexpected native companion for {target}: {name}")
            selected[name] = info
        elif (path.parts[:2] == ("clap_cpp", "licenses")
              or path.suffix.lower() in {".txt", ".license"}
              or path.name.upper().startswith(("LICENSE", "NOTICE", "COPYING"))):
            selected[name] = info
        else:
            raise ValueError(f"Cannot safely classify runtime companion: {name}")
    # Core license is optional; pack-node.py supplies the repository's LICENSE.
    # Preserve any wheel-provided additional license texts without dist-info metadata.
    for name, info in members.items():
        parts = PurePosixPath(name).parts
        if len(parts) >= 3 and parts[0].endswith(".dist-info") and parts[1] == "licenses" and not info.is_dir():
            distribution, separator, recorded_version = parts[0][:-10].rpartition("-")
            if not separator or distribution != "clap_cpp" or Version(recorded_version) != version:
                raise ValueError(f"Unexpected license metadata directory: {parts[0]}")
            relative = PurePosixPath(*parts[2:])
            destination = "clap_cpp/licenses/" + ("clap.txt" if str(relative) == "LICENSE" else f"wheel/{relative}")
            if destination in selected:
                raise ValueError(f"Duplicate runtime license destination: {destination}")
            selected[destination] = info
    return selected


def extract(wheels, output, target=None, version=None):
    requested = {target} if target else set(LIBRARIES)
    if not wheels.is_dir():
        raise ValueError(f"Wheel directory does not exist: {wheels}")
    expected = Version(version) if version is not None else None
    found = {}
    for wheel in sorted(wheels.rglob("*.whl")):
        distribution, wheel_version, _build, tags = parse_wheel_filename(wheel.name)
        if distribution in GPU_DISTRIBUTIONS:
            continue
        if distribution != "clap-cpp":
            raise ValueError(f"Unexpected wheel distribution: {distribution} ({wheel})")
        inferred = wheel_target(tags)
        if inferred not in requested:
            continue
        if inferred in found:
            raise ValueError(f"Duplicate wheel for {inferred}: {found[inferred][0]} and {wheel}")
        if expected is not None and wheel_version != expected:
            raise ValueError(f"Wheel version mismatch: {wheel.name}: {wheel_version} != {expected}")
        found[inferred] = (wheel, wheel_version)
    missing = requested - found.keys()
    if missing:
        raise ValueError(f"Missing CPU wheel targets: {', '.join(sorted(missing))}")
    if len({item[1] for item in found.values()}) != 1:
        raise ValueError("Wheel version mismatch across selected targets")
    # Refuse symlinked output ancestors and pre-existing targets, including dangling
    # symlinks. The temporary staging tree prevents malformed wheels from exposing
    # partial target directories. No archive path is passed to ZipFile.extract().
    for ancestor in (output, *output.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Unsafe symlink output ancestor: {ancestor}")
    for selected_target in requested:
        destination = output / selected_target
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Runtime target already exists: {destination}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-python-", dir=output) as temporary:
        staging = Path(temporary)
        for selected_target, (wheel, wheel_version) in sorted(found.items()):
            with zipfile.ZipFile(wheel) as archive:
                selected = runtime_members(archive, selected_target, wheel_version)
                for name, info in selected.items():
                    destination = staging / selected_target / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, destination.open("xb") as sink:
                        shutil.copyfileobj(source, sink)
                    # Native bytes and relative locations, including repaired RPATHs,
                    # are untouched; keep portable ordinary file permissions only.
                    destination.chmod(0o755 if NATIVE.search(name) else 0o644)
        for selected_target in sorted(found):
            (staging / selected_target).rename(output / selected_target)
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheels", type=Path, required=True, help="recursively search this directory for *.whl")
    parser.add_argument("--output", type=Path, required=True, help="runtime root, normally node/runtime")
    parser.add_argument("--target", choices=tuple(LIBRARIES), help="require only this target (ordinary CI)")
    parser.add_argument("--version", help="expected npm version, compared using packaging.version.Version")
    args = parser.parse_args()
    try:
        targets = extract(args.wheels, args.output, args.target, args.version)
    except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Extracted CPU wheel runtimes: {', '.join(targets)}")


if __name__ == "__main__":
    main()
