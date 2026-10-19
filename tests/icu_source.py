"""A bundled build must configure even when system ICU discovery is forbidden."""
import pathlib
import subprocess
import sys
import tempfile

source = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='clap-icu-configure-') as build:
    result = subprocess.run([
        sys.argv[1] if len(sys.argv) > 1 else 'cmake',
        '-S', str(source), '-B', build,
        '-DCLAP_USE_SYSTEM_ICU=OFF', '-DCMAKE_DISABLE_FIND_PACKAGE_ICU=TRUE',
        '-DBUILD_TESTING=OFF', '-DGGML_NATIVE=OFF', '-DGGML_CCACHE=OFF',
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    cache = (pathlib.Path(build) / 'CMakeCache.txt').read_text()
    assert 'CLAP_USE_SYSTEM_ICU:BOOL=OFF' in cache
    print('Bundled ICU configures without system ICU discovery')
