"""BF16 CLI text/waveform inference against independently decoded F32 storage."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave

from tiny_model import write_model

cli, decoder = map(Path, sys.argv[1:3])


def run(*args, check=True):
    return subprocess.run([str(cli), *map(str, args)], capture_output=True, text=True, check=check)


with tempfile.TemporaryDirectory(prefix="clap bf16 ") as tmp:
    root = Path(tmp)
    source, bf16, decoded = (root / name for name in ("source.gguf", "bf16.gguf", "decoded.gguf"))
    write_model(source, waveform=True)
    run("quantize", "--type", "bf16", source, bf16)
    subprocess.run([str(decoder), "decode", str(bf16), str(decoded)], check=True)
    assert bf16.stat().st_size < source.stat().st_size
    audio = root / "tone.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        wav.writeframes(b"".join(struct.pack("<h", round(8000 * math.sin(2 * math.pi * 440 * i / 48000)))
                                 for i in range(9600)))
    for args in [("--text", "A jazz quartet."), ("--audio", audio)]:
        actual = json.loads(run("embedding", "--model", bf16, *args).stdout)
        expected = json.loads(run("embedding", "--model", decoded, *args).stdout)
        assert len(actual) == len(expected) == 5
        assert all(math.isfinite(x) for x in actual)
        error = max(abs(x - y) for x, y in zip(actual, expected))
        assert error < 2e-6, (args[0], error, actual, expected)
    # BF16 is a floating source, including the identity storage conversion.
    repeated = root / "repeated.gguf"
    run("quantize", "--type", "BF16", bf16, repeated)
    assert repeated.read_bytes() == bf16.read_bytes()
    # A BF16 file is a floating Q8 source, not a block-quantized input.
    q8 = root / "q8.gguf"
    run("quantize", "--type", "Q8_0", bf16, q8)
    requantized = root / "requantized.gguf"
    rejected = run("quantize", "--type", "BF16", q8, requantized, check=False)
    assert rejected.returncode != 0 and "refusing requantization" in rejected.stderr
    assert not requantized.exists()
    # Do not replace destinations (including the source).
    for dest in (source, bf16):
        before = dest.read_bytes()
        assert run("quantize", "--type", "BF16", source, dest, check=False).returncode != 0
        assert dest.read_bytes() == before
    # Invalid/truncated floating input must never publish an output.
    truncated = root / "truncated.gguf"
    truncated.write_bytes(source.read_bytes()[:-32])
    invalid = root / "invalid.gguf"
    assert run("quantize", "--type", "BF16", truncated, invalid, check=False).returncode != 0
    assert not invalid.exists()
    assert not list(root.glob("*.tmp-*"))
print("BF16 CLI text/audio storage-rounded inference, identity and rejection passed")
