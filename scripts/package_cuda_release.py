#!/usr/bin/env python3
"""Package a statically linked clap backend and separate shared CUDA SDK runtimes.

Both archives have flat layouts: extract the matching cudart archive beside clap.
Only cudart/cuBLAS/cuBLASLt are redistributed, never the NVIDIA driver or stubs.
"""
import argparse
from pathlib import Path
import tarfile
import zipfile


def runtime_files(root, platform, arch):
    if platform == "windows":
        directories = [root / "bin" / arch]
        patterns = ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")
    else:
        directories = [root / "targets" / f"{arch}-linux" / "lib", root / "lib64", root / "lib"]
        patterns = ("libcudart.so.*", "libcublas.so.*", "libcublasLt.so.*")
    files = {}
    aliases = {}
    for pattern in patterns:
        matches = []
        for directory in directories:
            matches = sorted(directory.glob(pattern))
            if matches:
                break
        if not matches:
            raise ValueError(f"Missing required CUDA runtime {pattern} in {directories}")
        for path in matches:
            real = path.resolve(strict=True)
            if not real.is_file() or not real.stat().st_size:
                raise ValueError(f"Missing or empty CUDA runtime: {path}")
            files[real.name] = real
            if path.name != real.name:
                # Flatten toolkit symlink chains to portable, relative SONAME links.
                aliases[path.name] = real.name
    return files, aliases


def package(binary, asset, root, platform, arch, output):
    extension = ".zip" if platform == "windows" else ".tar.gz"
    if Path(asset).name != asset or not asset.endswith(extension):
        raise ValueError(f"Expected an asset filename ending in {extension}: {asset}")
    if not binary.is_file() or not binary.stat().st_size:
        raise ValueError(f"Missing or empty CLI executable: {binary}")
    files, aliases = runtime_files(root, platform, arch)
    output.mkdir(parents=True, exist_ok=True)
    executable_archive = output / asset
    runtime_archive = output / f"cudart-{asset}"
    if platform == "windows":
        with zipfile.ZipFile(executable_archive, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(binary, "clap.exe")
        with zipfile.ZipFile(runtime_archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, name)
    else:
        with tarfile.open(executable_archive, "w:gz") as archive:
            archive.add(binary, arcname="clap")
        with tarfile.open(runtime_archive, "w:gz", dereference=True) as archive:
            for name, path in sorted(files.items()):
                archive.add(path, arcname=name)
            for name, target in sorted(aliases.items()):
                entry = tarfile.TarInfo(name)
                entry.type = tarfile.SYMTYPE
                entry.linkname = target
                entry.mode = 0o777
                archive.addfile(entry)
    print(executable_archive)
    print(runtime_archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("linux", "windows"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/cuda"))
    args = parser.parse_args()
    try:
        package(args.binary, args.asset, args.cuda_root, args.platform, args.cuda_arch, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"CUDA release packaging failed: {error}\n")


if __name__ == "__main__":
    main()
