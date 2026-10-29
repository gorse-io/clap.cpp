#!/usr/bin/env python3
"""Stage pinned CPU sources for Cargo publication (Python standard library only).

Run from a recursive clap.cpp Git checkout before cargo package/publish. The
result is ignored by Git but explicitly included in the Cargo source archive.
Consumers of that archive do not need Python, Git, or network access to build.
"""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
SUBMODULES = ("ggml", "thirdparty/json", "thirdparty/icu", "thirdparty/miniaudio")
TREES = (
    "include", "src", "cmake/icu.cmake", "LICENSE",
    "ggml/CMakeLists.txt", "ggml/cmake", "ggml/include", "ggml/src", "ggml/LICENSE",
    "thirdparty/json/CMakeLists.txt", "thirdparty/json/cmake",
    "thirdparty/json/include", "thirdparty/json/nlohmann_json.natvis",
    "thirdparty/json/LICENSE.MIT", "thirdparty/miniaudio/miniaudio.h",
    "thirdparty/miniaudio/LICENSE", "thirdparty/icu/LICENSE",
    "thirdparty/icu/icu4c/source/common", "thirdparty/icu/icu4c/source/i18n",
    "thirdparty/icu/icu4c/source/stubdata",
)


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd or ROOT, text=True).strip()


def main():
    if not (ROOT / "include/clap.h").is_file():
        raise SystemExit("Run vendor.py from the rust directory of a clap.cpp Git checkout")
    revisions = {}
    for module in SUBMODULES:
        entry = git("ls-tree", "HEAD", "--", module).split()
        if len(entry) != 4 or entry[:2] != ["160000", "commit"]:
            raise SystemExit(f"{module}: expected a pinned Git submodule")
        directory = ROOT / module
        # rev-parse in an uninitialized directory otherwise returns the parent HEAD.
        if not (directory / ".git").exists():
            raise SystemExit(f"{module}: run git submodule update --init --recursive")
        actual = git("rev-parse", "HEAD", cwd=directory)
        if actual != entry[2]:
            raise SystemExit(f"{module}: expected pinned {entry[2]}, got {actual}")
        if git("status", "--porcelain", "--untracked-files=all", cwd=directory):
            raise SystemExit(f"{module}: refusing to vendor a dirty submodule")
        revisions[module] = actual
    for source in (*TREES, "python/tests/tiny_models.py"):
        if not (ROOT / source).exists():
            raise SystemExit(f"Missing {source}; run git submodule update --init --recursive")
    # Validate first, then replace only generated files. Never overwrite the bridge.
    with tempfile.TemporaryDirectory(prefix=".vendor-", dir=CRATE) as tmp:
        staged = Path(tmp) / "vendor"
        staged.mkdir()
        for source in TREES:
            src, dest = ROOT / source, staged / source
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest, ignore=shutil.ignore_patterns(
                    ".git", "build", "__pycache__", "*.pyc", "node_modules"))
            else:
                shutil.copy2(src, dest)
        (staged / "test-data").mkdir()
        shutil.copy2(ROOT / "python/tests/tiny_models.py", staged / "test-data/tiny_models.py")
        (staged / "revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")
        shutil.rmtree(CRATE / "vendor", ignore_errors=True)
        staged.rename(CRATE / "vendor")
    print("Vendored pinned CPU sources, upstream licenses, and stdlib test fixtures into rust/vendor")


if __name__ == "__main__":
    main()
