"""Native CLI contract (uses deterministic tiny-model fixture from parity.py)."""
import json
import pathlib
import subprocess
import sys
import wave

build, fixture = map(pathlib.Path, sys.argv[1:])
cli = build / 'clap'
reference = json.loads((fixture / 'fixture.json').read_text())
model = fixture / 'model.gguf'
for case in reference['text_cases']:
    result = subprocess.run([str(cli), 'embedding', '--model', str(model), '--text', case['text']], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    embedding = json.loads(result.stdout)
    assert len(embedding) == len(case['expected'])
    assert max(abs(a - b) for a, b in zip(embedding, case['expected'])) < 3e-5
for args in [[], ['--model', str(model)], ['--model', str(model), '--text', 'x', '--audio', 'x.wav']]:
    p = subprocess.run([str(cli), 'embedding', *args], capture_output=True)
    assert p.returncode != 0, args
wav = fixture / 'wrong-rate.wav'
with wave.open(str(wav), 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\0\0' * 100)
p = subprocess.run([str(cli), 'embedding', '--model', str(model), '--audio', str(wav)], capture_output=True, text=True)
# Resampled silence must behave exactly like native 48 kHz silence. Tiny parity
# models may use 32 mel bins and therefore intentionally lack a waveform frontend.
control_wav = fixture / 'native-rate.wav'
with wave.open(str(control_wav), 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000); w.writeframes(b'\0\0' * 300)
control = subprocess.run([str(cli), 'embedding', '--model', str(model), '--audio', str(control_wav)], capture_output=True, text=True)
assert (p.returncode, p.stdout, p.stderr) == (control.returncode, control.stdout, control.stderr)
if p.returncode:
    assert 'waveform frontend requires 64 mel bins' in p.stderr, p.stderr
else:
    assert len(json.loads(p.stdout)) == len(reference['text_cases'][0]['expected'])
print('Native model CLI contract passed')
