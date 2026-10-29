"""CPU integration tests: no downloads, mocks, NumPy or model dependencies."""
from array import array
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
from importlib.metadata import requires
import math
import struct
import wave

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
    with clap_cpp.CLAP(model_paths[0], threads=2) as result:
        yield result


@pytest.fixture
def mert(model_paths):
    with clap_cpp.CLAP(str(model_paths[1]), threads=2) as result:
        yield result


@pytest.fixture
def pcm():
    # One full second meets the MERT CLaMP3-feature minimum duration.
    return list(array("f", (0.2 * math.sin(2 * math.pi * 440 * i / 24000)
                            for i in range(24000))))


def assert_list(value, shape):
    assert isinstance(value, list)
    assert len(value) == shape[0]
    values = value
    if len(shape) == 2:
        assert all(isinstance(row, list) and len(row) == shape[1] for row in value)
        values = [element for row in value for element in row]
    assert all(type(element) is float and math.isfinite(element) for element in values)


def assert_close(actual, expected, rtol=1e-5, atol=1e-6):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        if isinstance(b, (list, tuple)):
            assert_close(a, b, rtol=rtol, atol=atol)
        else:
            assert abs(a - b) <= atol + rtol * abs(b)


def test_module_exports():
    assert clap_cpp.__all__ == ["CLAP", "backend"]
    assert clap_cpp.backend == "cpu"
    assert callable(clap_cpp.CLAP)
    for name in ("Model", "decode_audio", "resample_audio"):
        assert not hasattr(clap_cpp, name)
    for name in ("text", "audio"):
        assert not hasattr(clap_cpp.CLAP, name)


def test_package_has_no_numpy_dependency():
    assert all(not requirement.lower().startswith("numpy")
               for requirement in requires("clap-cpp") or [])


@pytest.mark.parametrize("as_string", [False, True])
def test_model_loads_from_unicode_path(tmp_path, as_string):
    directory = tmp_path / "模型 café 🎵"
    directory.mkdir()
    path = directory / "音声 modèle 🎶.gguf"
    clap_fixture(path)
    with clap_cpp.CLAP(str(path) if as_string else path, threads=2) as model:
        assert model.projection_dimension == 3
        tokens = model.tokenize("hello")
        assert_close(model.encode_text("hello"), text_reference(tokens), rtol=2e-4, atol=2e-5)


def test_properties_are_readonly(model, mert):
    assert model.projection_dimension == 3
    assert model.audio_sample_rate == 48000
    assert mert.projection_dimension == 4
    assert mert.audio_sample_rate == 24000
    for name in ("projection_dimension", "audio_sample_rate"):
        with pytest.raises(AttributeError):
            setattr(model, name, 99)


@pytest.mark.parametrize("text", ["hello", "world", "你好 café"])
def test_text_matches_independent_scalar_oracle(model, text):
    tokens = model.tokenize(text)
    assert isinstance(tokens, list)
    assert tokens and all(isinstance(token, int) for token in tokens)
    assert tokens[0] == 0 and tokens[-1] == 2
    actual = model.encode_text(text)
    assert_list(actual, (3,))
    assert_close(actual, text_reference(tokens), rtol=2e-4, atol=2e-5)
    assert_close(model.encode_text(tokens), actual)
    assert model.encode_text(text) == actual
    assert math.sqrt(sum(v * v for v in actual)) == pytest.approx(1, abs=1e-5)


def test_text_depends_on_input(model):
    assert max(abs(a - b) for a, b in zip(model.encode_text("hello"),
                                        model.encode_text("world"))) > 1e-7


def test_tokenizer_limit(model):
    assert len(model.tokenize("hello world" * 10, max_length=8)) <= 8


@pytest.mark.parametrize("container", [list, tuple, lambda x: array("f", x),
                                       lambda x: array("d", x),
                                       lambda x: memoryview(array("d", x))])
def test_waveform_inference_and_explicit_resampling(model, pcm, container):
    actual = model.encode_audio(pcm, sample_rate=24000)
    assert_list(actual, (3,))
    assert math.sqrt(sum(v * v for v in actual)) == pytest.approx(1, abs=1e-5)
    assert model.encode_audio(pcm, sample_rate=24000) == actual
    assert_close(model.encode_audio(container(pcm), sample_rate=24000), actual)
    # A strided double buffer must be converted safely, not read as packed F32.
    padded = array("d", (v for sample in pcm for v in (sample, 999)))
    assert_close(model.encode_audio(memoryview(padded)[::2], sample_rate=24000), actual)


def test_mert_frames_and_mean_pooling(mert, pcm):
    frames = mert.audio_frames(pcm, sample_rate=24000)
    assert_list(frames, (75, 4))
    mean = [sum(column) / len(frames) for column in zip(*frames)]
    assert_close(mert.encode_audio(pcm, sample_rate=24000), mean)
    assert mert.audio_frames(pcm, sample_rate=24000) == frames


def test_mert_clamp3_features(mert, pcm):
    features = mert.clamp3_features(pcm)
    assert_list(features, (1, 4))
    assert mert.clamp3_features(pcm) == features
    # Two retained 5-second windows; discard the final half-second tail.
    extended = (pcm * 11)[:252000]
    assert_list(mert.clamp3_features(extended), (2, 4))


@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("crop_offset", [0, 100])
def test_encode_audio_file_matches_pcm(model, pcm, tmp_path, as_string, crop_offset):
    path = tmp_path / "音声 café 🎶" / "立体声 modèle 🎵.wav"
    path.parent.mkdir()
    waveform = pcm * 11 if crop_offset else pcm
    samples = [(int(v * 32767), int(v * 0.5 * 32767)) for v in waveform]
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, 24000, len(samples), "NONE", "not compressed"))
        stream.writeframes(b"".join(struct.pack("<hh", *pair) for pair in samples))
    mono = [(left + right) / 65536 for left, right in samples]
    expected = model.encode_audio(mono, sample_rate=24000, crop_offset=crop_offset)
    actual = model.encode_audio(str(path) if as_string else path, crop_offset=crop_offset)
    assert_list(actual, (3,))
    assert_close(actual, expected)


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


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")], [1e100]])
@pytest.mark.parametrize("method", ["encode_audio", "audio_frames", "clamp3_features"])
def test_reject_invalid_pcm(mert, bad, method):
    with pytest.raises(ValueError):
        getattr(mert, method)(bad)


@pytest.mark.parametrize("bad", [[[0.0] * 3] * 2, ["invalid"], None, 1.0])
@pytest.mark.parametrize("method", ["encode_audio", "audio_frames", "clamp3_features"])
def test_reject_wrong_pcm_types(mert, bad, method):
    with pytest.raises(TypeError):
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
    model.close()
    with pytest.raises(RuntimeError):
        operation(model)


def test_context_manager_closes_on_exception(model_paths):
    with pytest.raises(LookupError):
        with clap_cpp.CLAP(model_paths[0], threads=1) as model:
            assert_list(model.encode_text("hello"), (3,))
            raise LookupError("body failed")
    with pytest.raises(RuntimeError):
        model.encode_text("hello")


def test_output_lists_own_their_lifetime(model_paths, pcm):
    with clap_cpp.CLAP(model_paths[0], threads=1) as model:
        text = model.encode_text("hello")
        saved = text.copy()
        model.encode_text("world")
        assert text == saved
    del model
    gc.collect()
    assert text == saved
    with clap_cpp.CLAP(model_paths[1], threads=1) as mert:
        frames = mert.audio_frames(pcm, sample_rate=24000)
        saved_frames = deepcopy(frames)
        features = mert.clamp3_features(pcm)
        saved_features = deepcopy(features)
    del mert
    gc.collect()
    assert frames == saved_frames
    assert features == saved_features
    frames[0][0] = 123.0
    assert frames[1] == saved_frames[1]
    assert features == saved_features


def test_same_handle_is_safe_under_concurrent_inference(model):
    prompts = ["hello", "world", "你好 café"]
    expected = {prompt: model.encode_text(prompt) for prompt in prompts}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [(prompt, executor.submit(model.encode_text, prompt)) for prompt in prompts * 4]
        for prompt, future in futures:
            assert_close(future.result(timeout=30), expected[prompt])
