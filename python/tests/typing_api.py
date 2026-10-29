"""Static-only consumer: checked by mypy, never executed."""

from pathlib import Path
from typing_extensions import assert_type

from clap_cpp import CLAP

model = CLAP(Path("model.gguf"), threads=2)
assert_type(CLAP("model.gguf", 2), CLAP)
assert_type(model.projection_dimension, int)
assert_type(model.audio_sample_rate, int)
assert_type(model.tokenize("music"), list[int])
assert_type(model.tokenize(text="music", max_length=32), list[int])
assert_type(model.encode_text("music"), list[float])
assert_type(model.encode_text(tokens=[1, 2]), list[float])
assert_type(model.encode_text((1, 2)), list[float])
assert_type(model.encode_audio("audio.wav"), list[float])
assert_type(model.encode_audio(path=Path("audio.wav"), crop_offset=3), list[float])
assert_type(model.encode_audio((0.0, 0.5)), list[float])
assert_type(model.encode_audio(pcm=[0.0, 0.5], sample_rate=24000, crop_offset=1), list[float])
assert_type(model.audio_frames([0.0], sample_rate=48000), list[list[float]])
assert_type(model.clamp3_features((0.0, 0.5)), list[list[float]])
assert_type(model.close(), None)
with CLAP("model.gguf") as opened:
    assert_type(opened, CLAP)

# Strict mypy rejects unused ignores, so these also catch accidental Any exports.
model.tokenize(123)  # type: ignore[arg-type]
model.encode_text(["not a token"])  # type: ignore[list-item]
model.encode_audio(path=Path("audio.wav"), sample_rate=48000)  # type: ignore[call-overload]
model.audio_frames([0.0], sample_rate="48000")  # type: ignore[arg-type]
model.audio_frames([[0.0]])  # type: ignore[list-item]
model.projection_dimension = 12  # type: ignore[misc]
CLAP("model.gguf", threads="2")  # type: ignore[arg-type]
