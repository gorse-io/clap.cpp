#!/usr/bin/env python3
"""Single-file CLaMP3 equals independent MERT + feature-input CLaMP3.
Run after mert_pipeline.py; all fixtures are generated, never downloaded.
"""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

build, out = map(Path, sys.argv[1:3])
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from conversion.base import write_gguf
from conversion.clamp3 import clamp3_inputs
from conversion.mert import mert_inputs
from mert_cache import install
from feature_input import encode

fixture = out / 'pipeline'
source = out / 'source'
tokenizer = build / 'clamp3-fixture/tokenizer.json'
revision = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
cli = str(build / 'clap-model')
install(out/'bundle-cache',source)

def run(model, *args, ok=True):
    result = subprocess.run([cli, '-m', str(model), *map(str, args), '-t', '2'],
                            capture_output=True, text=True)
    assert (result.returncode == 0) == ok, (args, result.returncode, result.stderr)
    return json.loads(result.stdout) if ok else result.stderr

for dtype in ['f32', 'f16']:
    standalone = fixture / f'clamp-{dtype}.gguf'
    bundled = fixture / f'bundled-{dtype}.gguf'
    base_args = [sys.executable, str(root/'convert_hf_to_gguf.py'), str(fixture/'model.pth'),
                 str(standalone), '--clamp3-config', str(fixture/'config.json'),
                 '--tokenizer', str(tokenizer), '--dtype', dtype]
    subprocess.run(base_args, check=True)
    bundled_args = base_args.copy(); bundled_args[3] = str(bundled)
    subprocess.run([*bundled_args, '--mert-revision', revision], check=True)
    # Native API verifies waveform dispatch, resampling, text and feature compatibility.
    subprocess.run([str(build/'clamp3-bundled-test'), str(bundled), str(standalone),
                    str(out/f'model-{dtype}.gguf')], check=True)
    for case in ['broadband', 'silence', 'tone', 'two_chunks', 'discard_tail', 'resampled-input']:
        wave = out / f'{case}.wav'
        features = run(out/f'model-{dtype}.gguf', '-a', wave, '--clamp3-features')
        expected = encode(build,standalone,features['values'],features['frames'],fixture)
        actual = run(bundled, '-a', wave)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        print(f'{dtype}/{case}: bundled vs dual max_abs={np.max(np.abs(np.asarray(actual)-expected))}', flush=True)
    np.testing.assert_array_equal(run(bundled, '--text', 'ab c'), run(standalone, '--text', 'ab c'))
    from q8_reference import tensors as stored_tensors
    original = {n:(t,v) for n,t,v in stored_tensors(bundled) if n.startswith('mert.')}
    for preset in ['Q8_0', 'Q4_K_M']:
        quantized = fixture / f'bundled-{dtype}-{preset}.gguf'
        if quantized.exists(): quantized.unlink()
        subprocess.run([str(build/'clap'), 'quantize', '--type', preset,
                        str(bundled), str(quantized)], check=True)
        rounded = {n:(t,v) for n,t,v in stored_tensors(quantized, build/'clap-q8-test' if preset=='Q4_K_M' else None)
                   if n.startswith('mert.')}
        assert original.keys()==rounded.keys()
        for name,(typ,values) in original.items():
            assert typ==rounded[name][0]
            np.testing.assert_array_equal(values,rounded[name][1])
        assert len(run(quantized, '-a', out/'broadband.wav'))==6
    for flags in [['--frames'], ['--clamp3-features'], ['--crop-offset', '0'],
                  ['--mert-model', out/f'model-{dtype}.gguf']]:
        run(bundled, '-a', out/'broadband.wav', *flags, ok=False)

# Corrupt one contract element at a time, keeping a structurally valid GGUF.
tensors, metadata = clamp3_inputs(fixture/'model.pth', fixture/'config.json', tokenizer, torch)
mert, mert_metadata = mert_inputs(source, revision, torch)
tensors += [('mert.'+name, value) for name, value in mert]
metadata.update({'clamp3.audio_frontend': 'mert-v1', 'clamp3.mert.config': mert_metadata['mert.config'],
                 'clamp3.mert.source_revision': revision})

def invalid(label, mutate, expected):
    ts, md = list(tensors), dict(metadata)
    mutate(ts, md)
    path = fixture/'invalid-bundled.gguf'
    write_gguf(path, ts, md, 'f32', np, torch)
    stderr = run(path, '--text', 'a', ok=False)
    assert expected in stderr, (label, stderr)
    print('rejected:', label, flush=True)

invalid('missing config', lambda ts, md: md.pop('clamp3.mert.config'), 'metadata')
invalid('missing revision', lambda ts, md: md.pop('clamp3.mert.source_revision'), 'metadata')
invalid('malformed revision', lambda ts, md: md.update({'clamp3.mert.source_revision': 'main'}), 'revision')
invalid('unsupported frontend', lambda ts, md: md.update({'clamp3.audio_frontend': 'mert-v2'}), 'frontend')
invalid('missing frontend marker', lambda ts, md: md.pop('clamp3.audio_frontend'), 'frontend')
invalid('missing tensor', lambda ts, md: ts.pop(), 'missing tensor')
invalid('unexpected frontend tensor', lambda ts, md: ts.append(('mert.unused.weight', torch.zeros(2))), 'unrecognized')
invalid('unexpected root tensor', lambda ts, md: ts.append(('unused.weight', torch.zeros(2))), 'unrecognized')
invalid('wrong shape', lambda ts, md: ts.__setitem__(-1, (ts[-1][0], torch.zeros(3, 7))), 'shape')

def bad_config(ts, md, key, value):
    config = json.loads(md['clamp3.mert.config']); config[key] = value
    md['clamp3.mert.config'] = json.dumps(config)

for key, value, error in [('hidden_size', 17, 'dimension'), ('num_hidden_layers', 1.5, 'num_hidden_layers'),
                           ('conv_stride', 'invalid', 'geometry'), ('deepnorm', True, 'variant')]:
    invalid(key, lambda ts, md, k=key, v=value: bad_config(ts, md, k, v), error)
# A valid MERT architecture incompatible with the CLaMP3 feature width fails before tensor loading.
invalid('feature width mismatch', lambda ts, md: bad_config(ts, md, 'hidden_size', 32), 'dimension')
# No silently ignored dependency flags for other converter architectures.
result = subprocess.run([sys.executable, str(root/'convert_hf_to_gguf.py'), str(source),
    str(fixture/'invalid-source.gguf'), '--mert-source', str(source), '--mert-revision', revision],
    capture_output=True, text=True)
assert result.returncode != 0 and 'unrecognized arguments' in result.stderr, result.stderr
print('Bundled CLaMP3 F32/F16 API/CLI/metadata validation passed', flush=True)
