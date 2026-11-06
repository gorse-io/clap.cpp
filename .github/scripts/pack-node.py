"""Stage npm optional platform packages from externally built shared runtimes."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    'linux-x64': 'libclap.so',
    'linux-arm64': 'libclap.so',
    'darwin-arm64': 'libclap.dylib',
    'win32-x64': 'clap.dll',
}


def stage(runtime_root, output, targets, root=ROOT):
    source = root / 'node'
    main = json.loads((source / 'package.json').read_text())
    version = main['version']
    main['optionalDependencies'] = {
        f'@gorse/clap-cpp-{target}': version for target in TARGETS
    }
    # Validate the complete requested input before replacing any staged package.
    for target in targets:
        runtime = runtime_root / target / 'clap_cpp'
        for name in [TARGETS[target], *[f'licenses/{name}.txt' for name in
                                      ['ggml', 'icu', 'json', 'miniaudio']]]:
            if not (runtime / name).is_file():
                raise ValueError(f'Missing {target} runtime input: {runtime / name}')
    output.mkdir(parents=True, exist_ok=True)
    for name in ['clap-cpp', *[f'clap-cpp-{target}' for target in TARGETS]]:
        destination = output / name
        if destination.is_symlink():
            raise ValueError(f'Refusing symlink package destination: {destination}')
        if destination.exists():
            shutil.rmtree(destination)
    package = output / 'clap-cpp'
    package.mkdir()
    for filename in ['index.js', 'index.d.ts', 'README.md']:
        shutil.copy2(source / filename, package / filename)
    shutil.copy2(root / 'LICENSE', package / 'LICENSE')
    (package / 'package.json').write_text(json.dumps(main, indent=2) + '\n')
    tests = package / 'test'
    tests.mkdir()
    for entry in (source / 'test').iterdir():
        if entry.is_file() and (entry.name.endswith('.test.js') or entry.name in
                               ['generate-fixtures.py', 'types.ts', 'tsconfig.json']):
            shutil.copy2(entry, tests / entry.name)
    shutil.copy2(root / 'python/tests/tiny_models.py', tests / 'tiny_models.py')
    for target in targets:
        package = output / f'clap-cpp-{target}'
        package.mkdir()
        runtime = package / 'runtime'
        # Retain the shared runtime's library-relative companion directories.
        shutil.copytree(runtime_root / target, runtime)
        shutil.copy2(root / 'LICENSE', runtime / 'clap_cpp/licenses/clap.txt')
        shutil.copy2(root / 'LICENSE', package / 'LICENSE')
        (package / 'README.md').write_text(
            f'# @gorse/clap-cpp-{target}\n\n'
            'Prebuilt shared runtime for clap-cpp. Installed automatically as an optional\n'
            'dependency on the matching platform. Use the clap-cpp package API.\n')
        os_name, cpu = target.split('-')
        manifest = {
            'name': f'@gorse/clap-cpp-{target}', 'version': version,
            'description': f'Prebuilt clap.cpp shared runtime for {target}',
            'license': main['license'], 'repository': main['repository'],
            'publishConfig': {'access': 'public'},
            'os': [os_name], 'cpu': [cpu], 'files': ['runtime', 'LICENSE', 'README.md'],
        }
        if os_name == 'linux':
            manifest['libc'] = ['glibc']
        (package / 'package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return [output / f'clap-cpp-{target}' for target in targets] + [output / 'clap-cpp']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=ROOT / 'node/runtime')
    parser.add_argument('--output', type=Path, default=ROOT / 'node/dist')
    parser.add_argument('--target', action='append', choices=TARGETS)
    args = parser.parse_args()
    for package in stage(args.runtime_root.resolve(), args.output.resolve(), args.target or list(TARGETS)):
        print(package)


if __name__ == '__main__':
    main()
