"""MERT2 full-song fixture adapter; pooled and frame numerical checks."""
import json

from model_contract import Command, Frames, Vector, float_wave


def cli(ctx):
    fixture = json.loads((ctx.fixture / "fixture.json").read_text())
    case = next(value for value in fixture["waveform_cases"] if value["full_song"])
    wav = ctx.fixture / "full-song.wav"
    float_wave(wav, fixture["sample_rate"], case["waveform"])
    args = ctx.embedding(ctx.fixture / "model.gguf", "--threads", "2", "--audio", wav)
    yield Command("full-song-pooled", args, timeout=120,
                  output=Vector(fixture["hidden_size"], case["expected_pooled"],
                                fixture["tolerance"], inclusive=True))
    yield Command("full-song-frames", [*args, "--frames"], timeout=120,
                  output=Frames(case["frames"], fixture["hidden_size"], case["expected_frames"], fixture["tolerance"]))
