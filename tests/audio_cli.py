"""Audio CLI validation runs without model weights or external codecs."""
from pathlib import Path
import subprocess
import sys

cli = sys.argv[1]
fixtures = Path(__file__).resolve().parent / "audio"
# The intentionally invalid model lets us check audio is decoded before loading.
model = fixtures / "invalid.bin"


def run(*args):
    return subprocess.run([cli, "--model", str(model), *args],
                          capture_output=True, text=True, timeout=10)


baseline = run("--text", "hello")
assert baseline.returncode != 0 and baseline.stderr
for name in ["mono.wav", "stereo.wav", "mono.mp3", "stereo.mp3", "mono.flac", "stereo.flac"]:
    result = run("--audio", str(fixtures / name))
    assert result.returncode != 0 and result.stderr == baseline.stderr, (name, result.stderr)
result = run("--audio", str(fixtures / "44100.flac"))
assert result.returncode != 0 and result.stderr == baseline.stderr, result.stderr
for name in ["empty.wav", "invalid.bin"]:
    result = run("--audio", str(fixtures / name))
    assert result.returncode != 0 and result.stderr != baseline.stderr, (name, result.stderr)
print("Audio CLI decoding and sample-rate validation passed")
