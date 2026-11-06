"""Validate npm package archives without installing or executing them."""
import argparse
import json
from pathlib import Path
import tarfile

TARGETS = {
    'linux-x64': 'libclap.so',
    'linux-arm64': 'libclap.so',
    'darwin-arm64': 'libclap.dylib',
    'win32-x64': 'clap.dll',
}


def validate_runtime(metadata, archive):
    target = metadata['name'].removeprefix('@gorse/clap-cpp-')
    if target not in TARGETS:
        raise ValueError(f'Unknown runtime target: {target}')
    os_name, cpu = target.split('-')
    if metadata.get('os') != [os_name] or metadata.get('cpu') != [cpu]:
        raise ValueError(f'Incorrect platform constraints: {target}')
    if metadata.get('libc') != (['glibc'] if os_name == 'linux' else None):
        raise ValueError(f'Incorrect libc constraint: {target}')
    required = ['LICENSE', f'runtime/clap_cpp/{TARGETS[target]}']
    required.extend(f'runtime/clap_cpp/licenses/{name}.txt'
                    for name in ['clap', 'ggml', 'icu', 'json', 'miniaudio'])
    with tarfile.open(archive) as contents:
        for name in required:
            try:
                entry = contents.getmember(f'package/{name}')
            except KeyError:
                raise ValueError(f'Missing {name} in {archive}') from None
            if not entry.isfile() or not entry.size:
                raise ValueError(f'Empty or invalid {name} in {archive}')


def manifest(archive):
    with tarfile.open(archive) as contents:
        source = contents.extractfile('package/package.json')
        if source is None:
            raise ValueError(f'Missing package manifest: {archive}')
        return json.load(source)


def verify(artifacts, require_all=False):
    main = None
    packages = {}
    for archive in artifacts.glob('*.tgz'):
        metadata = manifest(archive)
        if metadata['name'] == 'clap-cpp':
            if main is not None:
                raise ValueError('Multiple main tarballs')
            main = archive
        elif metadata['name'].startswith('@gorse/clap-cpp-'):
            if metadata['name'] in packages:
                raise ValueError(f'Duplicate runtime package: {metadata["name"]}')
            validate_runtime(metadata, archive)
            packages[metadata['name']] = metadata
    if main is None:
        raise ValueError('Missing main clap-cpp tarball')
    expected = {f'@gorse/clap-cpp-{target}' for target in TARGETS}
    if not packages or (require_all and set(packages) != expected):
        raise ValueError('Expected all four platform runtime archives' if require_all
                         else 'Missing platform runtime archive')
    main_manifest = manifest(main)
    version = main_manifest['version']
    if main_manifest.get('optionalDependencies') != {name: version for name in expected}:
        raise ValueError('Expected exact matching versions for all optional runtime packages')
    for name, metadata in packages.items():
        if metadata['version'] != version:
            raise ValueError(f'Mismatched runtime version: {name}')
    with tarfile.open(main) as contents:
        files = {entry.name for entry in contents if not entry.isdir()}
        required = {f'package/{name}' for name in
                    ['package.json', 'index.js', 'index.d.ts', 'README.md', 'LICENSE']}
        if files != required:
            raise ValueError(f'Unexpected main package contents: {files ^ required}')
    print(f'Validated main package and {len(packages)} platform runtime archive(s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    verify(args.artifacts.resolve(), args.require_all)
