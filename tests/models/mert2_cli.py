#!/usr/bin/env python3
"""Exercise MERT2 pooled and frame output through the public native CLI."""
import json
from pathlib import Path
import struct
import subprocess
import sys

binary, fixture_dir = Path(sys.argv[1]), Path(sys.argv[2])
fixture = json.loads((fixture_dir / "fixture.json").read_text())
case = next(value for value in fixture["waveform_cases"] if value["full_song"])
samples = struct.pack("<" + "f" * len(case["waveform"]), *case["waveform"])
wav = fixture_dir / "full-song.wav"
wav.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt " +
                struct.pack("<IHHIIHH", 16, 3, 1, fixture["sample_rate"],
                            fixture["sample_rate"] * 4, 4, 32) +
                b"data" + struct.pack("<I", len(samples)) + samples)

def run(*extra):
    result = subprocess.run([str(binary), "--model", str(fixture_dir / "model.gguf"),
                             "--threads", "2", "--audio", str(wav), *extra],
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)

pooled = run()
assert len(pooled) == fixture["hidden_size"]
assert max(abs(a - b) for a, b in zip(pooled, case["expected_pooled"])) <= fixture["tolerance"]
sequence = run("--frames")
assert sequence["frames"] == case["frames"]
assert sequence["dimension"] == fixture["hidden_size"]
assert len(sequence["values"]) == case["frames"] * fixture["hidden_size"]
assert max(abs(a - b) for a, b in zip(sequence["values"], case["expected_frames"])) <= fixture["tolerance"]
print("MERT2 CLI processed all", len(case["waveform"]) / fixture["sample_rate"],
      "seconds and returned", sequence["frames"], "frames")
