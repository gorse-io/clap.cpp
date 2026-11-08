"""LAION CLAP text parity and waveform resampling contract."""
import json

from model_contract import Command, Compare, Vector, pcm_silence


def cli(ctx):
    reference = json.loads((ctx.fixture / "fixture.json").read_text())
    model = ctx.fixture / "model.gguf"
    for index, case in enumerate(reference["text_cases"]):
        yield Command(f"text-{index}", ctx.embedding(model, "--text", case["text"]),
                      output=Vector(len(case["expected"]), case["expected"], 3e-5))
    for index, args in enumerate(([], ["--model", model],
                                  ["--model", model, "--text", "x", "--audio", "x.wav"])):
        yield Command(f"invalid-arguments-{index}", [ctx.cli, "embedding", *args], success=False)

    wav = ctx.fixture / "wrong-rate.wav"
    control_wav = ctx.fixture / "native-rate.wav"
    pcm_silence(wav, 16000, 100)
    pcm_silence(control_wav, 48000, 300)

    def resampling(results):
        # Tiny 32-mel models intentionally lack the waveform frontend. Compare
        # exit/stdout AND the stable error contract, never backend timing logs.
        actual, control = results
        assert (actual.returncode, actual.stdout) == (control.returncode, control.stdout), (
            f"resampled/native mismatch: {actual!r}\n{control!r}")
        if actual.returncode:
            for result in results:
                assert "waveform frontend requires 64 mel bins" in result.stderr, result.stderr
        else:
            Vector(len(reference["text_cases"][0]["expected"]))(json.loads(actual.stdout))

    yield Compare("waveform-resampling", [
        Command(str(path), ctx.embedding(model, "--audio", path), success=None)
        for path in (wav, control_wav)], resampling)
