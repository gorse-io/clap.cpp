"""Compare crate sources after Cargo repackaging, excluding generated metadata."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile

METADATA = {"Cargo.toml.orig", ".cargo_vcs_info.json"}


def sources(path):
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError(f"Unexpected non-file crate member: {member.name}")
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2 or ".." in parts or member.name.startswith("/"):
                raise ValueError(f"Invalid crate path: {member.name}")
            name = "/".join(parts[1:])
            if name in METADATA:
                continue
            if name in result:
                raise ValueError(f"Duplicate crate path: {name}")
            stream = archive.extractfile(member)
            assert stream is not None
            result[name] = stream.read()
    if "Cargo.toml" not in result or "vendor/include/clap.h" not in result:
        raise ValueError("Missing package manifest or vendored native sources")
    return result


def verify(original, repacked):
    before, after = sources(original), sources(repacked)
    changed = {name for name in before.keys() | after.keys()
               if before.get(name) != after.get(name)}
    if changed:
        raise ValueError(f"Repackaged crate sources differ: {sorted(changed)}")
    print(f"Verified source equivalence for {len(before)} crate files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("repacked", type=Path)
    args = parser.parse_args()
    verify(args.original, args.repacked)
