"""CLaMP3 fixture data; waveform-only CLI and malformed config contracts."""
import json

from model_contract import Command, Vector, metadata_model, pcm_silence, read_metadata_model


def cli(ctx):
    path = ctx.fixture / "input.wav"
    pcm_silence(path, 48000, 48000)
    model = ctx.fixture / "model.gguf"
    yield Command("waveform", ctx.embedding(model, "-a", path, "-t", "2"), output=Vector(6))
    for name, args in (
            ("obsolete-audio-features", ["--audio-features", "obsolete.npy"]),
            ("obsolete-mert-model", ["--mert-model", model]),
            ("obsolete-crop-offset", ["-a", path, "--crop-offset", "0"])):
        yield Command(name, ctx.embedding(model, *args), success=False, nonempty_stderr=True)
    yield Command("bad-architecture", ctx.embedding(ctx.fixture / "bad-architecture.gguf", "--text", "a"),
                  success=False, diagnostics=("architecture",))


def invalid(ctx):
    metadata, tensors, infos, payload = read_metadata_model(ctx.fixture / "model.gguf")
    for name, branch, key, value, expected in (
            ("negative-pad", "text_config", "pad_token_id", -1, "padding token"),
            ("negative-epsilon", "audio_config", "layer_norm_eps", -1, "layer norm epsilon"),
            ("causal-decoder", "text_config", "is_decoder", True, "decoder")):
        config = json.loads(metadata["clamp3.config"])
        config[branch][key] = value
        changed = dict(metadata, **{"clamp3.config": json.dumps(config)})
        path = ctx.fixture / (name + ".gguf")
        metadata_model(path, changed, tensors, infos, payload)
        yield Command(name, ctx.embedding(path, "--text", "ab"), success=False, diagnostics=(expected,))
