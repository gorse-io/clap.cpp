#!/usr/bin/env python3
"""Opt-in published MuQ checkpoint vs pinned official inference components.

Checks the published weight SHA and both executed custom-source hashes before
inference. No downloads, pickle loading, or training/RVQ execution are used.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import torch
from safetensors import safe_open
from muq_reference import InferenceCore, MODEL_REVISION, SOURCE_REVISION

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('build', type=Path)
parser.add_argument('checkpoint', type=Path)
parser.add_argument('reference', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--dtype', choices=['f32', 'f16'], default='f32')
parser.add_argument('--gguf', type=Path)
parser.add_argument('--tone-diagnostic', action='store_true',
                    help='reproduce pure-tone near-noise-floor FFT sensitivity (same strict tolerance)')
args = parser.parse_args()
checkpoint = args.checkpoint / 'model.safetensors'
digest = hashlib.sha256()
with checkpoint.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
assert digest.hexdigest() == '273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6'
raw = json.loads((args.checkpoint / 'config.json').read_text())
assert raw['encoder_dim'] == 1024 and raw['encoder_depth'] == 12 and raw['conv_dim'] == 512
torch.set_num_threads(2)
with torch.device('meta'):
    model = InferenceCore(raw, args.reference)
with safe_open(checkpoint, framework='pt', device='cpu') as source:
    state = {}
    for name in source.keys():
        if name.startswith(('model.conv.', 'model.conformer.layers.', 'model.conformer.layer_norm.',
                            'model.conformer.embed_positions.')):
            value = source.get_tensor(name)
            if args.dtype == 'f16' and value.ndim > 1:
                value = value.half().float()
            state[name.removeprefix('model.')] = value
loaded = model.load_state_dict(state, strict=False, assign=True)
assert not loaded.unexpected_keys, loaded
assert all(name.startswith('conformer.pos_conv_embed.') for name in loaded.missing_keys), loaded
model.eval()
time = torch.arange(5041, dtype=torch.float32) / 24000
wave = 0.17 * torch.sin(2 * torch.pi * 220 * time) + 0.09 * torch.cos(2 * torch.pi * 733 * time)
wave[-1] += 0.31
if not args.tone_diagnostic:
    # Seeded low-level broadband energy keeps this strict architecture oracle
    # away from the log-floor conditioning issue. Keep --tone-diagnostic as a
    # separate, reproducible strict check; never loosen its tolerance silently.
    wave += 0.002 * torch.randn(wave.shape, generator=torch.Generator().manual_seed(20261021))
with torch.inference_mode():
    expected = model(wave[None])[0].float()
assert expected.shape == (6, 1024) and torch.isfinite(expected).all()
expected_frames, expected_pooled = expected.flatten().tolist(), expected.mean(0).tolist()
del expected, model, state
gc.collect()
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'expected.json').write_text(json.dumps({'waveform': wave.tolist(),
    'frames': expected_frames, 'pooled': expected_pooled, 'tone_diagnostic': args.tone_diagnostic}))
gguf = args.gguf or args.output / ('muq-' + args.dtype + '.gguf')
if args.gguf is None:
    converter = Path(__file__).resolve().parents[2] / 'convert_hf_to_gguf.py'
    subprocess.run([sys.executable, str(converter), str(args.checkpoint), str(gguf),
                    '--dtype', args.dtype, '--muq-revision', MODEL_REVISION], check=True)
data = struct.pack('<' + 'f' * len(wave), *wave.tolist())
wav = args.output / 'published.wav'
wav.write_bytes(b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' +
                struct.pack('<IHHIIHH', 16, 3, 1, 24000, 96000, 4, 32) +
                b'data' + struct.pack('<I', len(data)) + data)
command = [str(args.build / 'clap-model'), '--model', str(gguf), '--threads', '2', '--audio', str(wav)]
summary = {'model_revision': MODEL_REVISION, 'source_revision': SOURCE_REVISION,
           'samples': len(wave), 'sample_rate': 24000, 'frames': 6, 'dimension': 1024,
           'dtype': args.dtype, 'tone_diagnostic': args.tone_diagnostic,
           'reference': 'official components, storage-rounded F32 activations'}
for frames, expected in [(True, expected_frames), (False, expected_pooled)]:
    result = subprocess.run(command + (['--frames'] if frames else []),
                            check=True, text=True, capture_output=True, timeout=600)
    actual = json.loads(result.stdout)
    if frames:
        assert actual['frames'] == 6 and actual['dimension'] == 1024
        actual = actual['values']
    assert len(actual) == len(expected) and all(math.isfinite(v) for v in actual)
    error = max(abs(a - b) for a, b in zip(actual, expected))
    kind = 'frame' if frames else 'pooled'
    summary[kind + '_max_abs_error'] = error
    summary[kind + '_rms_error'] = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, expected)) / len(actual))
    summary[kind + '_cosine'] = sum(a * b for a, b in zip(actual, expected)) / math.sqrt(
        sum(a * a for a in actual) * sum(b * b for b in expected))
    (args.output / ('native-' + kind + '.json')).write_text(json.dumps(actual))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    assert error <= (5e-4 if args.dtype == 'f32' else 3e-3), (args.dtype, frames, error)
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
