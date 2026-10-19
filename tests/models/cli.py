"""Native CLI contract (uses deterministic tiny-model fixture from parity.py)."""
import json
import pathlib
import subprocess
import sys
import wave

build, fixture = map(pathlib.Path, sys.argv[1:])
cli = build / 'clap-model'
reference = json.loads((fixture / 'fixture.json').read_text())
model = fixture / 'model.gguf'
for case in reference['text_cases']:
    result = subprocess.run([str(cli), '--model', str(model), '--text', case['text']], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    embedding = json.loads(result.stdout)
    assert len(embedding) == len(case['expected'])
    assert max(abs(a - b) for a, b in zip(embedding, case['expected'])) < 3e-5
for args in [[], ['--model', str(model)], ['--model', str(model), '--text', 'x', '--audio', 'x.wav']]:
    p = subprocess.run([str(cli), *args], capture_output=True)
    assert p.returncode != 0, args
wav = fixture / 'wrong-rate.wav'
with wave.open(str(wav), 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\0\0' * 100)
p = subprocess.run([str(cli), '--model', str(model), '--audio', str(wav)], capture_output=True, text=True)
assert p.returncode != 0 and '48000' in p.stderr, p.stderr
print('Native model CLI contract passed')
