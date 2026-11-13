"""Same-instance lifecycle and concurrency tests; no downloads or NumPy.

Run with a staged package on PYTHONPATH (the native payload stays package-local).
The barrier proves Python dispatch overlap, not native graph execution overlap.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import math
from pathlib import Path
import threading

import pytest

binding = pytest.importorskip("clap_cpp.clap", exc_type=ImportError)
CLAP = binding.CLAP


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "concurrency_tiny_model", Path(__file__).resolve().parents[2] / "tests" / "tiny_model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path_factory.mktemp("concurrent-model") / "tiny.gguf"
    return path, module.write_model(path)


def outcome(call):
    try:
        return call()
    except (ValueError, RuntimeError) as exc:
        # The reduced-mel model intentionally cannot run the waveform frontend;
        # the real ABI's errors still exercise leases and TLS diagnostics.
        return type(exc), str(exc)


@pytest.mark.parametrize("name,invoke", [
    ("clap_encode_text", lambda m: m.encode_text("hello")),
    ("clap_encode_tokens", lambda m: m.encode_text(tokens=[0, 109, 106, 113, 113, 116, 2])),
    ("clap_encode_audio", lambda m: m.encode_audio([0.0] * 32)),
    ("clap_encode_audio_file", lambda m: m.encode_audio("missing-concurrency.wav")),
    ("clap_get_audio_frames", lambda m: m.audio_frames([0.0] * 32)),
    ("clap_tokenize", lambda m: m.tokenize("hello")),
    ("clap_projection_dimension", lambda m: m.projection_dimension),
])
def test_same_instance_calls_overlap_at_native_boundary(tiny_model, monkeypatch, name, invoke):
    with CLAP(tiny_model[0]) as model:
        expected = outcome(lambda: invoke(model))
        actual = getattr(binding._lib, name)
        rendezvous = threading.Barrier(2, timeout=3)

        def overlapping(*args):
            rendezvous.wait()
            return actual(*args)

        monkeypatch.setattr(binding._lib, name, overlapping)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [pool.submit(outcome, lambda: invoke(model)) for _ in range(2)]
            assert [future.result(timeout=15) for future in results] == [expected, expected]


def test_concurrent_real_inference_matches_scalar_oracle(tiny_model):
    path, metadata = tiny_model
    cases = metadata["cases"] * 3
    def infer(index_case):
        index, case = index_case
        if index % 2:
            return model.encode_text(tokens=model.tokenize(case["input"]))
        return model.encode_text(case["input"])

    with CLAP(path) as model, ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(infer, enumerate(cases)))
    for result, case in zip(results, cases):
        assert isinstance(result, list)
        assert result == pytest.approx(case["embedding"], abs=2e-5)


def test_same_instance_mixed_text_and_waveform_succeed(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAP_DEVICE", "cpu")
    spec = importlib.util.spec_from_file_location(
        "waveform_concurrency_fixture", Path(__file__).resolve().parents[2] / "tests" / "tiny_model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "waveform.gguf"
    metadata = module.write_model(path, waveform=True)
    samples = [0.1 * math.sin(2 * math.pi * 440 * i / 48000) for i in range(48000)]
    with CLAP(path) as model:
        expected_audio = model.encode_audio(samples, 48000)
        with ThreadPoolExecutor(max_workers=2) as pool:
            text = pool.submit(model.encode_text, metadata["cases"][0]["input"])
            audio = pool.submit(model.encode_audio, samples, 48000)
            actual_text, actual_audio = text.result(timeout=30), audio.result(timeout=30)
    assert actual_text == pytest.approx(metadata["cases"][0]["embedding"], abs=2e-5)
    assert actual_audio == pytest.approx(expected_audio, abs=2e-5)


@pytest.mark.parametrize("call_count", [1, 2])
@pytest.mark.parametrize("reentrant", [False, True])
def test_close_rejects_new_calls_and_defers_destroy_until_inflight_returns(
        tiny_model, monkeypatch, call_count, reentrant):
    model = CLAP(tiny_model[0])
    actual_encode = binding._lib.clap_encode_text
    actual_destroy = binding._lib.clap_destroy
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    rendezvous = threading.Barrier(call_count, timeout=5)
    destroyed = []

    def blocked(*args):
        rendezvous.wait()
        entered.set()
        assert release.wait(10)
        assert not destroyed
        return actual_encode(*args)

    def destroy(handle):
        destroyed.append(handle.value)
        actual_destroy(handle)

    def close():
        if reentrant:
            class Token:
                def __index__(self):
                    model.close()
                    return 0

            with pytest.raises(RuntimeError, match="closed"):
                model.encode_text(tokens=[Token()])
        else:
            model.close()
        closed.set()

    monkeypatch.setattr(binding._lib, "clap_encode_text", blocked)
    monkeypatch.setattr(binding._lib, "clap_destroy", destroy)
    with ThreadPoolExecutor(max_workers=call_count + 1) as pool:
        pending = [pool.submit(model.encode_text, "hello") for _ in range(call_count)]
        try:
            assert entered.wait(5)
            closing = pool.submit(close)
            assert closed.wait(2), "close must defer destruction rather than block an active call"
            assert not destroyed
            with pytest.raises(RuntimeError, match="closed"):
                model.projection_dimension
            model.close()
        finally:
            release.set()
        for future in pending:
            assert future.result(timeout=10) == pytest.approx(tiny_model[1]["cases"][0]["embedding"], abs=2e-5)
        closing.result(timeout=10)
    assert len(destroyed) == 1
    model.close()
    assert len(destroyed) == 1


@pytest.mark.parametrize("kind", ["text", "token", "pcm", "path", "rate"])
def test_user_conversion_can_close_before_lease(tiny_model, kind):
    with CLAP(tiny_model[0]) as model:
        class Text(str):
            def encode(self, *args, **kwargs):
                model.close()
                return super().encode(*args, **kwargs)

        class Integer:
            def __index__(self):
                model.close()
                return 48000 if kind == "rate" else 0

        class Sample:
            def __float__(self):
                model.close()
                return 0.0

        class File:
            def __fspath__(self):
                model.close()
                return "missing.wav"

        calls = {
            "text": lambda: model.encode_text(Text("hello")),
            "token": lambda: model.encode_text([Integer()]),
            "pcm": lambda: model.audio_frames([Sample()]),
            "path": lambda: model.encode_audio(File()),
            "rate": lambda: model.encode_audio([0.0], sample_rate=Integer()),
        }
        with pytest.raises(RuntimeError, match="closed"):
            calls[kind]()
