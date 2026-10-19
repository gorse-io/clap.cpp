"""Canonical quantizer/native Q8 integration; standard library only."""
import os
import pathlib
import subprocess
import sys
import tempfile

runner, quantizer = map(pathlib.Path, sys.argv[1:3])
regression = sys.argv[3] if len(sys.argv) > 3 else None
with tempfile.TemporaryDirectory(prefix='clap q8 ') as tmp:
    out = pathlib.Path(tmp)
    source, canonical = out / 'f32.gguf', out / 'canonical.gguf'
    subprocess.run([str(runner), 'create', str(source), str(canonical)], check=True)
    assert quantizer.is_file(), 'clap-quantize executable missing (Q8 conversion unsupported)'
    result = out / 'converted.gguf'
    if regression == 'alignment':
        source = pathlib.Path(str(source) + '.align64.gguf')
        subprocess.run([str(quantizer), str(source), str(result)], check=True)
        subprocess.run([str(runner), 'verify', str(source), str(result)], check=True)
        print('64-byte input alignment roundtrip passed')
        sys.exit(0)
    if regression == 'symlink':
        target = out / 'absent.gguf'
        try:
            result.symlink_to(target)
        except (OSError, NotImplementedError) as error:
            print(f'Symlink creation unavailable: {error}')
            sys.exit(77)
        p = subprocess.run([str(quantizer), str(source), str(result)], capture_output=True)
        assert p.returncode != 0, 'quantizer replaced a dangling destination symlink'
        assert result.is_symlink() and result.readlink() == target
        assert not target.exists()
        assert not list(out.glob('*.tmp-*')), 'temporary output leaked'
        print('Dangling destination symlink preserved')
        sys.exit(0)
    if regression == 'concurrent':
        env = os.environ.copy()
        env['LD_PRELOAD'] = str(pathlib.Path(sys.argv[4]).resolve())
        env['CLAP_QUANTIZE_RACE_DESTINATION'] = str(result)
        p = subprocess.run([str(quantizer), str(source), str(result)], env=env, capture_output=True)
        assert p.returncode != 0, 'quantizer overwrote a concurrently created destination'
        assert result.read_bytes() == b'concurrent destination\n'
        assert not list(out.glob('*.tmp-*')), 'temporary output leaked after publication failure'
        print('Deterministic publication race preserved concurrent destination')
        sys.exit(0)
    assert regression is None, f'unknown regression: {regression}'
    subprocess.run([str(quantizer), str(source), str(result)], check=True)
    subprocess.run([str(runner), 'verify', str(source), str(result)], check=True)
    assert result.stat().st_size < source.stat().st_size
    half_source = pathlib.Path(str(source) + '.f16.gguf')
    half_result = out / 'from-f16.gguf'
    subprocess.run([str(quantizer), str(half_source), str(half_result)], check=True)
    subprocess.run([str(runner), 'verify', str(half_source), str(half_result)], check=True)
    # Requantization, clobbering and malformed input must fail without writing.
    for src, dest in [(source, source), (source, result), (result, out / 'requantized.gguf')]:
        before = dest.read_bytes() if dest.exists() else None
        p = subprocess.run([str(quantizer), str(src), str(dest)], capture_output=True)
        assert p.returncode != 0
        assert (dest.read_bytes() if dest.exists() else None) == before
    bad = out / 'truncated.gguf'
    bad.write_bytes(source.read_bytes()[:-32])
    p = subprocess.run([str(quantizer), str(bad), str(out / 'invalid.gguf')], capture_output=True)
    assert p.returncode != 0 and not (out / 'invalid.gguf').exists()
    unsupported = out / 'unsupported.gguf'
    unsupported.write_bytes(source.read_bytes().replace(b'clap', b'nope', 1))
    for src, dest in [(unsupported, out / 'unsupported-output.gguf'),
                      (source, out / 'missing-directory' / 'output.gguf')]:
        p = subprocess.run([str(quantizer), str(src), str(dest)], capture_output=True)
        assert p.returncode != 0 and not dest.exists()
    assert not list(out.glob('*.tmp-*')), 'temporary output leaked on failure'
print('Q8 quantizer/native integration and rejection tests passed')
