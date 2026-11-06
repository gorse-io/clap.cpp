"""Generate native inference fixtures using existing stdlib-only generators."""
import importlib.util
from pathlib import Path
import struct
import sys
import wave

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tiny_models", root / "python/tests/tiny_models.py")
assert spec is not None and spec.loader is not None
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "java/target/fixtures"
out.mkdir(parents=True, exist_ok=True)
tiny.clap_fixture(out / "tiny-clap.gguf")
tiny.mert_fixture(out / "tiny-mert.gguf")
# Independent scalar oracle, with the tokenizer's 512-position padded input.
ids = [0] + [b + 5 for b in "Hello, 世界 🎵".encode("utf-8")] + [2]
ids.extend([1] * (512 - len(ids)))
(out / "text-reference.txt").write_text("\n".join(map(str, tiny.text_reference(ids))), encoding="utf-8")
# Exactly representable PCM, so Java float[] and WAV decoder inputs match.
values = [((i % 97) - 48) * 128 for i in range(24000)]
with wave.open(str(out / "audio 世界.wav"), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(24000)
    wav.writeframes(struct.pack("<" + "h" * len(values), *values))
(out / "bad.wav").write_bytes(b"not an audio file")
