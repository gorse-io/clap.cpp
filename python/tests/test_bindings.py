"""CPU integration tests: no downloads, mocks, or optional model dependencies."""
from concurrent.futures import ThreadPoolExecutor
import gc
import wave

import numpy as np
import pytest

import clap_cpp
from tiny_models import clap_fixture, mert_fixture, text_reference, write_gguf


@pytest.fixture(scope="session")
def model_paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-models")
    clap_path, mert_path = root / "clap.gguf", root / "mert.gguf"
    clap_fixture(clap_path)
    mert_fixture(mert_path)
    return clap_path, mert_path


@pytest.fixture
def model(model_paths):
    with clap_cpp.CLAP(model_paths[0], threads=2) as result:
        yield result


@pytest.fixture
def mert(model_paths):
    with clap_cpp.CLAP(str(model_paths[1]), threads=2) as result:
        yield result


@pytest.fixture
def pcm():
    # A full second also meets the MERT CLaMP3-feature minimum duration.
    t = np.arange(24000, dtype=np.float64) / 24000
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def assert_array(value, shape):
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float32
    assert value.shape == shape
    assert np.isfinite(value).all()


def test_module_exports():
    assert clap_cpp.__all__ == ["CLAP"]
    assert callable(clap_cpp.CLAP)
    for name in ("Model", "decode_audio", "resample_audio"):
        assert not hasattr(clap_cpp, name)
    for name in ("text", "audio"):
        assert not hasattr(clap_cpp.CLAP, name)


@pytest.mark.parametrize("as_string", [False, True])
def test_model_loads_from_unicode_path(tmp_path, as_string):
    directory = tmp_path / "模型 café 🎵"
    directory.mkdir()
    path = directory / "音声 modèle 🎶.gguf"
    clap_fixture(path)
    with clap_cpp.CLAP(str(path) if as_string else path, threads=2) as model:
        assert model.projection_dimension == 3
        tokens = model.tokenize("hello")
        np.testing.assert_allclose(model.encode_text("hello"), text_reference(tokens),
                                   rtol=2e-4, atol=2e-5)


def test_optional_mmproj_preserves_existing_loading(model_paths):
    with clap_cpp.CLAP(model_paths[0], 1, mmproj=None) as model:
        assert_array(model.encode_text("hello"), (3,))


@pytest.mark.parametrize("as_string", [False, True])
def test_explicit_mmproj_is_not_ignored(model_paths, tmp_path, as_string):
    missing = tmp_path / "missing-audio-mmproj.gguf"
    # A metadata-only dispatch probe: failure must name the projector, not
    # the incomplete model weights. This tests the exact forwarded path.
    probe = tmp_path / "embeddinggemma2-dispatch.gguf"
    write_gguf(probe, {"general.architecture": "gemma-embedding2"}, {})
    with pytest.raises(RuntimeError, match="missing-audio-mmproj"):
        clap_cpp.CLAP(probe, threads=1,
                      mmproj=str(missing) if as_string else missing)
    # Other architectures must not silently ignore a supplied projector either.
    with pytest.raises(RuntimeError):
        clap_cpp.CLAP(model_paths[0], mmproj=missing)


def test_properties_are_readonly(model, mert):
    assert model.projection_dimension == 3
    assert model.audio_sample_rate == 48000
    assert mert.projection_dimension == 4
    assert mert.audio_sample_rate == 24000
    for name in ("projection_dimension", "audio_sample_rate"):
        with pytest.raises(AttributeError):
            setattr(model, name, 99)


@pytest.mark.parametrize("text", ["hello", "world", "你好 café"])
def test_text_matches_independent_numpy_oracle(model, text):
    tokens = model.tokenize(text)
    assert isinstance(tokens, list)
    assert tokens and all(isinstance(token, int) for token in tokens)
    assert tokens[0] == 0 and tokens[-1] == 2
    actual = model.encode_text(text)
    assert_array(actual, (3,))
    np.testing.assert_allclose(actual, text_reference(tokens), rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(model.encode_text(tokens), actual, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(model.encode_text(text), actual)
    assert np.linalg.norm(actual) == pytest.approx(1, abs=1e-5)


def test_text_depends_on_input(model):
    assert np.max(np.abs(model.encode_text("hello") - model.encode_text("world"))) > 1e-7


def test_tokenizer_limit(model):
    assert len(model.tokenize("hello world" * 10, max_length=8)) <= 8


def test_waveform_inference_and_explicit_resampling(model, pcm):
    # Exercise the actual 64-mel waveform frontend, not feature-only tensors.
    actual = model.encode_audio(pcm, sample_rate=24000)
    assert_array(actual, (3,))
    assert np.linalg.norm(actual) == pytest.approx(1, abs=1e-5)
    np.testing.assert_array_equal(model.encode_audio(pcm, sample_rate=24000), actual)
    np.testing.assert_allclose(model.encode_audio(pcm.tolist(), sample_rate=24000), actual,
                               rtol=1e-5, atol=1e-6)
    # Non-contiguous float64 input must be converted safely, not read as F32.
    padded = np.empty((pcm.size, 2), dtype=np.float64)
    padded[:, 0] = pcm
    padded[:, 1] = 999
    np.testing.assert_allclose(model.encode_audio(padded[:, 0], sample_rate=24000), actual,
                               rtol=1e-5, atol=1e-6)


def test_mert_frames_and_mean_pooling(mert, pcm):
    frames = mert.audio_frames(pcm, sample_rate=24000)
    assert_array(frames, (75, 4))
    np.testing.assert_allclose(mert.encode_audio(pcm, sample_rate=24000), frames.mean(axis=0),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(mert.audio_frames(pcm, sample_rate=24000), frames)


def test_mert_clamp3_features(mert, pcm):
    features = mert.clamp3_features(pcm)
    assert_array(features, (1, 4))
    np.testing.assert_array_equal(mert.clamp3_features(pcm), features)
    # Two retained 5-second windows; discard the final half-second tail.
    extended = np.tile(pcm, 11)[:252000]
    assert_array(mert.clamp3_features(extended), (2, 4))


@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("crop_offset", [0, 100])
def test_encode_audio_file_matches_pcm(model, pcm, tmp_path, as_string, crop_offset):
    path = tmp_path / "音声 café 🎶" / "立体声 modèle 🎵.wav"
    path.parent.mkdir()
    waveform = np.tile(pcm, 11) if crop_offset else pcm
    samples = np.column_stack((waveform, waveform * 0.5))
    samples = (samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, 24000, len(samples), "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())
    mono = samples.astype(np.float32).mean(axis=1) / 32768
    expected = model.encode_audio(mono, sample_rate=24000, crop_offset=crop_offset)
    actual = model.encode_audio(str(path) if as_string else path, crop_offset=crop_offset)
    assert_array(actual, (3,))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_encode_audio_file_errors(model, tmp_path):
    missing = tmp_path / "missing.wav"
    with pytest.raises(RuntimeError):
        model.encode_audio(missing)
    invalid = tmp_path / "invalid.wav"
    invalid.write_text("not audio")
    with pytest.raises(RuntimeError):
        model.encode_audio(invalid)
    with pytest.raises(ValueError):
        model.encode_audio(missing, crop_offset=-1)
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.encode_audio(missing)


@pytest.mark.parametrize("bad", [[], np.zeros((2, 3)), [float("nan")], [float("inf")]])
@pytest.mark.parametrize("method", ["encode_audio", "audio_frames", "clamp3_features"])
def test_reject_invalid_pcm(mert, bad, method):
    with pytest.raises(ValueError):
        getattr(mert, method)(bad)


@pytest.mark.parametrize("rate", [0, -1])
@pytest.mark.parametrize("method", ["encode_audio", "audio_frames", "clamp3_features"])
def test_model_rejects_nonpositive_rates(mert, pcm, rate, method):
    with pytest.raises(ValueError):
        getattr(mert, method)(pcm, sample_rate=rate)


def test_rejects_negative_options(model, model_paths, pcm):
    with pytest.raises(ValueError):
        clap_cpp.CLAP(model_paths[0], threads=-1)
    with pytest.raises(ValueError):
        model.tokenize("hello", max_length=-1)
    with pytest.raises(ValueError):
        model.encode_audio(pcm, crop_offset=-1)


@pytest.mark.parametrize("bad", [None, 123, object()])
def test_text_rejects_wrong_types(model, bad):
    with pytest.raises(TypeError):
        model.encode_text(bad)


def test_native_failures_are_runtime_errors(model, mert, pcm, tmp_path):
    with pytest.raises(RuntimeError):
        clap_cpp.CLAP(tmp_path / "missing.gguf")
    with pytest.raises(RuntimeError):
        model.encode_audio(tmp_path / "missing.wav")
    invalid = tmp_path / "not-a-model.gguf"
    invalid.write_text("not a GGUF")
    with pytest.raises(RuntimeError):
        clap_cpp.CLAP(invalid)
    with pytest.raises(RuntimeError):
        model.audio_frames(pcm)
    with pytest.raises(RuntimeError):
        model.clamp3_features(pcm)
    with pytest.raises(RuntimeError):
        mert.encode_text("audio-only model")
    with pytest.raises(RuntimeError):
        mert.clamp3_features(pcm[:100])


@pytest.mark.parametrize("operation", [
    lambda m: m.projection_dimension,
    lambda m: m.audio_sample_rate,
    lambda m: m.tokenize("hello"),
    lambda m: m.encode_text("hello"),
    lambda m: m.encode_text([0, 2]),
    lambda m: m.encode_audio([0.1, 0.2]),
    lambda m: m.audio_frames([0.1, 0.2]),
    lambda m: m.clamp3_features([0.1, 0.2]),
])
def test_closed_model_rejects_every_operation(model_paths, operation):
    model = clap_cpp.CLAP(model_paths[0], threads=1)
    model.close()
    model.close()  # Idempotent resource release.
    with pytest.raises(RuntimeError):
        operation(model)


def test_context_manager_closes_on_exception(model_paths):
    with pytest.raises(LookupError):
        with clap_cpp.CLAP(model_paths[0], threads=1) as model:
            assert_array(model.encode_text("hello"), (3,))
            raise LookupError("body failed")
    with pytest.raises(RuntimeError):
        model.encode_text("hello")


def test_output_arrays_own_their_lifetime(model_paths, pcm):
    with clap_cpp.CLAP(model_paths[0], threads=1) as model:
        text = model.encode_text("hello")
        saved = text.copy()
        model.encode_text("world")
        np.testing.assert_array_equal(text, saved)
    del model
    gc.collect()
    np.testing.assert_array_equal(text, saved)
    with clap_cpp.CLAP(model_paths[1], threads=1) as mert:
        frames = mert.audio_frames(pcm, sample_rate=24000)
        saved_frames = frames.copy()
        features = mert.clamp3_features(pcm)
        saved_features = features.copy()
    del mert
    gc.collect()
    np.testing.assert_array_equal(frames, saved_frames)
    np.testing.assert_array_equal(features, saved_features)


def test_same_handle_is_safe_under_concurrent_inference(model):
    prompts = ["hello", "world", "你好 café"]
    expected = {prompt: model.encode_text(prompt) for prompt in prompts}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [(prompt, executor.submit(model.encode_text, prompt)) for prompt in prompts * 4]
        for prompt, future in futures:
            np.testing.assert_allclose(future.result(timeout=30), expected[prompt],
                                       rtol=1e-5, atol=1e-6)
