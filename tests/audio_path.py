"""Exercise the C++ file-path API using generated waveform-capable weights."""
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

from tiny_model import write_model

with tempfile.TemporaryDirectory(prefix="clap-audio-path-") as directory:
    model = Path(directory) / "tiny-waveform.gguf"
    write_model(model, waveform=True)
    long_audio = Path(directory) / "long clip.wav"
    with wave.open(str(Path(sys.argv[2]) / "mono.wav"), "rb") as source:
        params = source.getparams()
        samples = source.readframes(source.getnframes())
    with wave.open(str(long_audio), "wb") as output:
        output.setparams(params)
        output.writeframes(samples * 110)
    subprocess.run([sys.argv[1], str(model), sys.argv[2], str(long_audio)], check=True, timeout=90)
