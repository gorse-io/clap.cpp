"""Generate local tiny GGUF/WAV fixtures; stdlib only, no downloads."""
import ast
import importlib.util
import math
from pathlib import Path
import struct
import sys
import wave

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tiny_models", root / "python/tests/tiny_models.py")
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)
# Reuse the independent scalar oracle without executing Rust's generator.
tree = ast.parse((root / "rust/tests/generate.py").read_text())
reference = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "text_reference")
namespace = {"math": math, "struct": struct}
exec(compile(ast.Module(body=[reference], type_ignores=[]), "text-reference", "exec"), namespace)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "go/build/fixtures"
out.mkdir(parents=True, exist_ok=True)
tiny.clap_fixture(out / "模型.gguf")
tiny.mert_fixture(out / "mert.gguf")
(out / "text-reference.txt").write_text(" ".join(map(str, namespace["text_reference"]([0, 77, 110, 2]))))
def wav(name, rate, samples, channels=1):
    with wave.open(str(out / name), "wb") as f:
        f.setparams((channels, 2, rate, len(samples) // channels, "NONE", "not compressed"))
        f.writeframes(struct.pack("<" + "h" * len(samples), *samples))
wav("音频.wav", 24000, [0, 8192, -16384, 24576])
wav("立体声.wav", 24000, [-4096,4096,4096,12288,-20480,-12288,20480,28672], 2)
samples = [int(8192 * math.sin(i * .03)) for i in range(12 * 48000)]
wav("long.wav", 48000, samples)
wav("crop.wav", 48000, samples[48000:11 * 48000])
wav("mert.wav", 24000, [int(6553 * math.sin(i * .03)) for i in range(24000)])
(out / "invalid.wav").write_bytes(b"not an audio file")
