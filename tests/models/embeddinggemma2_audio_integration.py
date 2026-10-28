"""Generated end-to-end audio API/CLI oracle, no downloads or ML packages."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave
import embeddinggemma2 as text
import embeddinggemma2_audio as audio


def run(binary, cli):
    with tempfile.TemporaryDirectory(prefix="gemma-audio-integration-") as directory:
        root = Path(directory)
        ints = [int(4000 * math.sin(i * .173) + 2000 * math.cos(i * .091)) for i in range(1601)]
        pcm = [v / 32768 for v in ints]
        wav = root / "input.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(struct.pack("<" + "h" * len(ints), *ints))
        for storage in ("f32", "f16"):
            metadata, tensors = audio.fixture(storage, projection=8)
            soft = audio.oracle(audio.frontend(pcm), tensors)
            text_metadata, text_tensors, expected = text.fixture(storage, audio_soft=soft)
            model, mmproj, oracle = [root / (storage + suffix) for suffix in (".gguf", "-mmproj.gguf", ".json")]
            text.save(model, text_metadata, text_tensors, storage)
            audio.save(mmproj, metadata, tensors, storage)
            oracle.write_text(json.dumps({"pcm": pcm, "embedding": expected["audio_embedding"]}), encoding="utf-8")
            subprocess.run([binary, str(model), str(mmproj), str(oracle)], check=True, timeout=30)
            command = [cli, "embedding", "--model", str(model), "--mmproj", str(mmproj),
                       "--threads", "2", "--audio", str(wav)]
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            actual = json.loads(result.stdout)
            assert len(actual) == len(expected["audio_embedding"])
            assert all(math.isfinite(v) for v in actual)
            assert max(abs(a-b) for a,b in zip(actual, expected["audio_embedding"])) < 5e-5
        print("F32/F16 native waveform API, resampling and WAV CLI match independent oracle")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
