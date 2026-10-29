"""Set Rust/npm package versions from a shared SemVer release tag."""
import argparse
import json
import os
from pathlib import Path
import re

SEMVER = re.compile(
    r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)


def release_version(tag):
    match = SEMVER.fullmatch(tag)
    if not match:
        raise ValueError(f"Expected a SemVer release tag, got {tag!r}")
    prerelease = match.group(4)
    if prerelease and any(p.isdigit() and len(p) > 1 and p.startswith("0")
                          for p in prerelease.split(".")):
        raise ValueError("Numeric prerelease identifiers must not have leading zeroes")
    # npm strips build metadata; reject it instead of publishing different versions.
    if match.group(5):
        raise ValueError("Release tags with build metadata are not supported by npm")
    return tag.removeprefix("v")


def update(root, tag):
    version = release_version(tag)
    manifest = root / "rust/Cargo.toml"
    text = manifest.read_text()
    text, count = re.subn(r'^version = "[^"]+"$', f'version = "{version}"', text,
                          count=0, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Expected exactly one Rust package version")
    files = {}
    for name in ("node/package.json", "node/package-lock.json"):
        path = root / name
        data = json.loads(path.read_text())
        data["version"] = version
        if name.endswith("package-lock.json"):
            data["packages"][""]["version"] = version
        files[path] = json.dumps(data, indent=2) + "\n"
    manifest.write_text(text)
    for path, content in files.items():
        path.write_text(content)
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    args = parser.parse_args()
    version = update(Path(__file__).resolve().parents[2], args.tag)
    print(version)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"version={version}\n")
            output.write(f"npm-tag={'next' if '-' in version else 'latest'}\n")
