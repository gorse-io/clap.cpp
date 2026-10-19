#!/usr/bin/env python3
import json
import math
import wave
from pathlib import Path
import subprocess
import struct
import sys

binary, fixture = Path(sys.argv[1]), Path(sys.argv[2])
reference = json.loads((fixture / 'fixture.json').read_text())

def run(*args, ok=True):
    result = subprocess.run([str(binary), '--model', str(fixture / 'model.gguf'), *args],
                            text=True, capture_output=True)
    if ok and result.returncode:
        raise RuntimeError(result.stderr)
    if not ok and result.returncode == 0:
        raise RuntimeError('command unexpectedly succeeded')
    return result

actual = json.loads(run('--audio-features', str(fixture / 'short-audio.npy')).stdout)
expected = reference['audio_cases'][0]['expected']
assert len(actual) == len(expected) and all(math.isfinite(value) for value in actual)
assert max(abs(a - b) for a, b in zip(actual, expected)) < 8e-5
frames = reference['audio_cases'][0]['frames']
width = len(reference['audio_cases'][0]['features']) // frames
payload = struct.pack('<' + 'f' * (frames * width), *reference['audio_cases'][0]['features'])
for shape in [f'(-{frames}, {width})', f'({frames}.0, {width})', f'({frames}, -{width})']:
    header = ("{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }\n").encode()
    invalid = fixture / 'invalid-shape.npy'
    invalid.write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + payload)
    result = run('--audio-features', str(invalid), ok=False)
    assert 'shape' in result.stderr, result.stderr
wav = fixture / 'input.wav'
with wave.open(str(wav), 'wb') as output:
    output.setnchannels(1); output.setsampwidth(2); output.setframerate(48000)
    output.writeframes(b'\0\0' * 480)
result = run('--audio', str(wav), ok=False)
assert 'MERT-v1-95M' in result.stderr
result = subprocess.run([str(binary), '--model', str(fixture / 'bad-architecture.gguf'), '--text', 'a'],
                        text=True, capture_output=True)
assert result.returncode != 0 and 'architecture' in result.stderr
print('CLaMP3 .npy CLI and invalid-model rejection passed')
