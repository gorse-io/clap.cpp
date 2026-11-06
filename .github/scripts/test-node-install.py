"""Exercise npm's optional platform selection against actual staged tarballs.

Only @gorse packages use the temporary local registry; Koffi comes from the normal
registry. This is a packaging test, not publication to the public npm registry.
"""
import argparse
import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from urllib.parse import unquote, urlparse

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


@contextmanager
def registry(packages):
    downloads = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            assert isinstance(self.server, ThreadingHTTPServer)
            name = unquote(urlparse(self.path).path).lstrip('/')
            if name.startswith('tarballs/'):
                filename = name.removeprefix('tarballs/')
                matches = [(m, p) for m, p in packages.values() if p.name == filename]
                if not matches:
                    self.send_error(404)
                    return
                metadata, archive = matches[0]
                downloads.append(metadata['name'])
                payload = archive.read_bytes()
            elif name in packages:
                metadata, archive = packages[name]
                version = metadata['version']
                data = archive.read_bytes()
                version_data = dict(metadata, dist={
                    'tarball': f'http://127.0.0.1:{self.server.server_address[1]}/tarballs/{archive.name}',
                    'shasum': hashlib.sha1(data).hexdigest(),
                    'integrity': 'sha512-' + base64.b64encode(hashlib.sha512(data).digest()).decode(),
                })
                payload = json.dumps({'name': name, 'dist-tags': {'latest': version},
                                      'versions': {version: version_data}}).encode()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream' if name.startswith('tarballs/') else 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', downloads
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def verify(artifacts, prefix, require_all=False):
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
            packages[metadata['name']] = metadata, archive
            validate_runtime(metadata, archive)
    if main is None:
        raise ValueError('Missing main clap-cpp tarball')
    if require_all and set(packages) != {f'@gorse/clap-cpp-{target}' for target in TARGETS}:
        raise ValueError('Expected all four platform runtime archives')
    host = subprocess.check_output(['node', '-p', 'process.platform + "-" + process.arch'], text=True).strip()
    expected = f'@gorse/clap-cpp-{host}'
    if expected not in packages:
        raise ValueError(f'Missing real host runtime archive: {expected}')
    main_manifest = manifest(main)
    for name, (metadata, _) in packages.items():
        if main_manifest['optionalDependencies'].get(name) != metadata['version']:
            raise ValueError(f'Unpinned/mismatched optional runtime version: {name}')
    prefix.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='clap-npm-registry-') as scratch, registry(packages) as (url, downloads):
        config = Path(scratch) / '.npmrc'
        config.write_text(f'@gorse:registry={url}\n')
        command = [shutil.which('npm'), 'install', '--prefix', str(prefix),
                   '--userconfig', str(config), '--cache', str(Path(scratch) / 'cache'),
                   '--ignore-scripts', '--no-audit', '--no-fund', str(main)]
        subprocess.run(command, check=True, shell=sys.platform == 'win32')
        if downloads != [expected]:
            raise ValueError(f'Expected only {expected} tarball download, got {downloads}')
    installed = prefix / 'node_modules'
    for name in main_manifest['optionalDependencies']:
        if (installed / name).exists() != (name == expected):
            raise ValueError(f'Incorrect installed optional dependency: {name}')
    subprocess.run(['node', '-e', 'const m = require(process.argv[1]); if (typeof m.CLAP !== "function") process.exit(1)',
                    str(installed / 'clap-cpp')], check=True)
    print(f'Only the host runtime was downloaded and installed: {expected}; native library loaded successfully')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    verify(args.artifacts.resolve(), args.prefix.resolve(), args.require_all)
