"""CPU integration tests: no downloads, mocks, or optional model dependencies."""
from concurrent.futures import ThreadPoolExecutor
import gc
import math
import wave

import numpy as np
import pytest

import clap_cpp
from tiny_models import clap_fixture, mert_fixture, text_reference


@pytest.fixture(scope="session")
def model_paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-models")
    clap_path, mert_path = root / "clap.gguf", root / "mert.gguf"
    clap_fixture(clap_path)
    mert_fixture(mert_path)
    return clap_path, mert_path


@pytest.fixture
def model(model_paths):
    with clap_cpp.Model(model_paths[0], threads=2) as result:
        yield result


@pytest.fixture
def mert(model_paths):
    with clap_cpp.Model(str(model_paths[1]), threads=2) as result:
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
    assert callable(clap_cpp.Model)
    assert callable(clap_cpp.decode_audio)
    assert callable(clap_cpp.resample_audio)


@pytest.mark.parametrize("as_string", [False, True])
def test_model_loads_from_unicode_path(tmp_path, as_string):
    directory = tmp_path / "模型 café 🎵"
    directory.mkdir()
    path = directory / "音声 modèle 🎶.gguf"
    clap_fixture(path)
    with clap_cpp.Model(str(path) if as_string else path, threads=2) as model:
        assert model.projection_dimension == 3
        tokens = model.tokenize("hello")
        np.testing.assert_allclose(model.text("hello"), text_reference(tokens),
                                   rtol=2e-4, atol=2e-5)


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
    actual = model.text(text)
    assert_array(actual, (3,))
    np.testing.assert_allclose(actual, text_reference(tokens), rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(model.text(tokens), actual, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(model.text(text), actual)
    assert np.linalg.norm(actual) == pytest.approx(1, abs=1e-5)


def test_text_depends_on_input(model):
    assert np.max(np.abs(model.text("hello") - model.text("world"))) > 1e-7


def test_tokenizer_limit(model):
    assert len(model.tokenize("hello world" * 10, max_length=8)) <= 8


def test_waveform_inference_and_explicit_resampling(model, pcm):
    # Exercise the actual 64-mel waveform frontend, not feature-only tensors.
    actual = model.audio(pcm, sample_rate=24000)
    assert_array(actual, (3,))
    assert np.linalg.norm(actual) == pytest.approx(1, abs=1e-5)
    np.testing.assert_array_equal(model.audio(pcm, sample_rate=24000), actual)
    np.testing.assert_allclose(model.audio(pcm.tolist(), sample_rate=24000), actual,
                               rtol=1e-5, atol=1e-6)
    upsampled = clap_cpp.resample_audio(pcm, 24000, 48000)
    np.testing.assert_allclose(model.audio(upsampled), actual, rtol=1e-5, atol=1e-6)
    # Non-contiguous float64 input must be converted safely, not read as F32.
    padded = np.empty((pcm.size, 2), dtype=np.float64)
    padded[:, 0] = pcm
    padded[:, 1] = 999
    np.testing.assert_allclose(model.audio(padded[:, 0], sample_rate=24000), actual,
                               rtol=1e-5, atol=1e-6)


def test_mert_frames_and_mean_pooling(mert, pcm):
    frames = mert.audio_frames(pcm, sample_rate=24000)
    assert_array(frames, (75, 4))
    np.testing.assert_allclose(mert.audio(pcm, sample_rate=24000), frames.mean(axis=0),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(mert.audio_frames(pcm, sample_rate=24000), frames)


def test_mert_clamp3_features(mert, pcm):
    features = mert.clamp3_features(pcm)
    assert_array(features, (1, 4))
    np.testing.assert_array_equal(mert.clamp3_features(pcm), features)
    # Two retained 5-second windows; discard the final half-second tail.
    extended = np.tile(pcm, 11)[:252000]
    assert_array(mert.clamp3_features(extended), (2, 4))


@pytest.mark.parametrize("name", ["stereo input.wav", "音声 café 🎶/立体声 modèle 🎵.wav"])
def test_decoded_stereo_wav_preserves_rate_and_averages_channels(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.array([[32767, -32768], [16384, 0], [-16384, 8192], [0, 0]],
                       dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, 22050, len(samples), "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())
    decoded, rate = clap_cpp.decode_audio(path)
    assert rate == 22050
    assert_array(decoded, (4,))
    np.testing.assert_allclose(decoded, samples.astype(np.float32).mean(axis=1) / 32768,
                               rtol=0, atol=2e-5)
    decoded_str, rate_str = clap_cpp.decode_audio(str(path))
    assert rate_str == rate
    np.testing.assert_array_equal(decoded_str, decoded)


def test_decoded_audio_is_accepted_by_model(model, pcm, tmp_path):
    path = tmp_path / "tone.wav"
    samples = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 24000, len(samples), "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())
    decoded, rate = clap_cpp.decode_audio(path)
    assert rate == 24000
    np.testing.assert_allclose(decoded, samples.astype(np.float32) / 32768,
                               rtol=0, atol=2e-5)
    assert_array(model.audio(decoded, sample_rate=rate), (3,))


def test_resampling_identity_duration_and_list_input(pcm):
    np.testing.assert_array_equal(clap_cpp.resample_audio(pcm, 24000, 24000), pcm)
    np.testing.assert_array_equal(clap_cpp.resample_audio(pcm.tolist(), 24000, 24000), pcm)
    assert_array(clap_cpp.resample_audio(pcm, 24000), (48000,))
    assert_array(clap_cpp.resample_audio(pcm[:101], 24000, 16000), (math.ceil(101 * 2 / 3),))


@pytest.mark.parametrize("bad", [[], np.zeros((2, 3)), [float("nan")], [float("inf")]])
@pytest.mark.parametrize("method", ["audio", "audio_frames", "clamp3_features"])
def test_reject_invalid_pcm(mert, bad, method):
    with pytest.raises(ValueError):
        getattr(mert, method)(bad)


@pytest.mark.parametrize("bad", [[], np.zeros((2, 3)), [float("nan")], [float("inf")]])
def test_resample_rejects_invalid_pcm(bad):
    with pytest.raises(ValueError):
        clap_cpp.resample_audio(bad, 24000)


@pytest.mark.parametrize("rate", [0, -1])
@pytest.mark.parametrize("method", ["audio", "audio_frames", "clamp3_features"])
def test_model_rejects_nonpositive_rates(mert, pcm, rate, method):
    with pytest.raises(ValueError):
        getattr(mert, method)(pcm, sample_rate=rate)


@pytest.mark.parametrize("source,target", [(0, 48000), (-1, 48000), (24000, 0), (24000, -1)])
def test_resample_rejects_nonpositive_rates(pcm, source, target):
    with pytest.raises(ValueError):
        clap_cpp.resample_audio(pcm, source, target)


def test_rejects_negative_options(model, model_paths, pcm):
    with pytest.raises(ValueError):
        clap_cpp.Model(model_paths[0], threads=-1)
    with pytest.raises(ValueError):
        model.tokenize("hello", max_length=-1)
    with pytest.raises(ValueError):
        model.audio(pcm, crop_offset=-1)


@pytest.mark.parametrize("bad", [None, 123, object()])
def test_text_rejects_wrong_types(model, bad):
    with pytest.raises(TypeError):
        model.text(bad)


def test_native_failures_are_runtime_errors(model, mert, pcm, tmp_path):
    with pytest.raises(RuntimeError):
        clap_cpp.Model(tmp_path / "missing.gguf")
    with pytest.raises(RuntimeError):
        clap_cpp.decode_audio(tmp_path / "missing.wav")
    invalid = tmp_path / "not-a-model.gguf"
    invalid.write_text("not a GGUF")
    with pytest.raises(RuntimeError):
        clap_cpp.Model(invalid)
    with pytest.raises(RuntimeError):
        model.audio_frames(pcm)
    with pytest.raises(RuntimeError):
        model.clamp3_features(pcm)
    with pytest.raises(RuntimeError):
        mert.text("audio-only model")
    with pytest.raises(RuntimeError):
        mert.clamp3_features(pcm[:100])


@pytest.mark.parametrize("operation", [
    lambda m: m.projection_dimension,
    lambda m: m.audio_sample_rate,
    lambda m: m.tokenize("hello"),
    lambda m: m.text("hello"),
    lambda m: m.text([0, 2]),
    lambda m: m.audio([0.1, 0.2]),
    lambda m: m.audio_frames([0.1, 0.2]),
    lambda m: m.clamp3_features([0.1, 0.2]),
])
def test_closed_model_rejects_every_operation(model_paths, operation):
    model = clap_cpp.Model(model_paths[0], threads=1)
    model.close()
    model.close()  # Idempotent resource release.
    with pytest.raises(RuntimeError):
        operation(model)


def test_context_manager_closes_on_exception(model_paths):
    with pytest.raises(LookupError):
        with clap_cpp.Model(model_paths[0], threads=1) as model:
            assert_array(model.text("hello"), (3,))
            raise LookupError("body failed")
    with pytest.raises(RuntimeError):
        model.text("hello")


def test_output_arrays_own_their_lifetime(model_paths, pcm):
    with clap_cpp.Model(model_paths[0], threads=1) as model:
        text = model.text("hello")
        saved = text.copy()
        model.text("world")
        np.testing.assert_array_equal(text, saved)
    del model
    gc.collect()
    np.testing.assert_array_equal(text, saved)
    with clap_cpp.Model(model_paths[1], threads=1) as mert:
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
    expected = {prompt: model.text(prompt) for prompt in prompts}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [(prompt, executor.submit(model.text, prompt)) for prompt in prompts * 4]
        for prompt, future in futures:
            np.testing.assert_allclose(future.result(timeout=30), expected[prompt],
                                       rtol=1e-5, atol=1e-6)
