"""CPU integration tests: no downloads, mocks, NumPy or model dependencies."""
from array import array
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
import weakref
import gc
from importlib.metadata import requires
import math
import struct
import wave

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
    # One second of mono PCM for waveform inference and frame pooling.
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


@pytest.mark.parametrize("copier", [copy, deepcopy])
@pytest.mark.parametrize("explicit_close", [False, True])
def test_model_rejects_copy_without_transferring_ownership(model_paths, monkeypatch, copier, explicit_close):
    from clap_cpp import clap as wrapper

    destroy = wrapper._lib.clap_destroy
    destroyed = []

    def record_destroy(handle):
        destroyed.append(handle.value)
        destroy(handle)

    monkeypatch.setattr(wrapper._lib, "clap_destroy", record_destroy)
    original = clap_cpp.CLAP(model_paths[0], threads=2)
    reference = weakref.ref(original)
    expected = original.encode_text("hello")
    try:
        with pytest.raises(TypeError, match="CLAP.*cannot be copied"):
            duplicate = copier(original)
            # If this regresses, disarm the unsafe duplicate before failing:
            # two finalizers on a real handle would abort the test process.
            duplicate._handle = type(duplicate._handle)()
        gc.collect()
        assert destroyed == []
        assert original.encode_text("hello") == expected
        if explicit_close:
            original.close()
            original.close()
            original.__del__()
            assert len(destroyed) == 1
    finally:
        del original
        gc.collect()
    assert reference() is None
    assert len(destroyed) == 1


def test_module_exports():
    assert clap_cpp.__all__ == ["CLAP", "backend"]
    assert clap_cpp.backend == "cpu"
    assert callable(clap_cpp.CLAP)
    for name in ("Model", "decode_audio", "resample_audio"):
        assert not hasattr(clap_cpp, name)
    for name in ("text", "audio", "clamp3_features"):
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


def test_optional_mmproj_preserves_existing_loading(model_paths):
    with clap_cpp.CLAP(model_paths[0], 1, mmproj=None) as model:
        assert_list(model.encode_text("hello"), (3,))


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
@pytest.mark.parametrize("method", ["encode_audio", "audio_frames"])
def test_reject_invalid_pcm(mert, bad, method):
    with pytest.raises(ValueError):
        getattr(mert, method)(bad)


@pytest.mark.parametrize("bad", [[[0.0] * 3] * 2, ["invalid"], None, 1.0])
@pytest.mark.parametrize("method", ["encode_audio", "audio_frames"])
def test_reject_wrong_pcm_types(mert, bad, method):
    with pytest.raises(TypeError):
        getattr(mert, method)(bad)


@pytest.mark.parametrize("rate", [0, -1])
@pytest.mark.parametrize("method", ["encode_audio", "audio_frames"])
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
        mert.encode_text("audio-only model")


@pytest.mark.parametrize("operation", [
    lambda m: m.projection_dimension,
    lambda m: m.audio_sample_rate,
    lambda m: m.tokenize("hello"),
    lambda m: m.encode_text("hello"),
    lambda m: m.encode_text([0, 2]),
    lambda m: m.encode_audio([0.1, 0.2]),
    lambda m: m.audio_frames([0.1, 0.2]),
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
    del mert
    gc.collect()
    assert frames == saved_frames
    frames[0][0] = 123.0
    assert frames[1] == saved_frames[1]


def test_same_handle_is_safe_under_concurrent_inference(model):
    prompts = ["hello", "world", "你好 café"]
    expected = {prompt: model.encode_text(prompt) for prompt in prompts}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [(prompt, executor.submit(model.encode_text, prompt)) for prompt in prompts * 4]
        for prompt, future in futures:
            assert_close(future.result(timeout=30), expected[prompt])


@pytest.mark.parametrize("value", [2**31, 2**64, 2**100])
def test_rejects_native_integer_overflow(model, model_paths, value):
    with pytest.raises(OverflowError):
        clap_cpp.CLAP(model_paths[0], threads=value)
    with pytest.raises(OverflowError):
        model.encode_audio([0.1], sample_rate=value)
    with pytest.raises(OverflowError):
        model.encode_text([value])
    with pytest.raises(OverflowError):
        model.encode_text([-value - 1])


@pytest.mark.parametrize("value", [1.5, "1", None])
def test_rejects_noninteger_abi_arguments(model, model_paths, value):
    with pytest.raises(TypeError):
        clap_cpp.CLAP(model_paths[0], threads=value)
    with pytest.raises(TypeError):
        model.tokenize("hello", max_length=value)
    with pytest.raises(TypeError):
        model.encode_audio([0.1], sample_rate=value)
    with pytest.raises(TypeError):
        model.encode_audio([0.1], crop_offset=value)
    with pytest.raises(TypeError):
        model.encode_text([value])


def test_size_t_overflow_is_not_truncated(model):
    with pytest.raises(OverflowError):
        model.tokenize("hello", max_length=2**128)
    with pytest.raises(OverflowError):
        model.encode_audio([0.1], crop_offset=2**128)


def test_nul_rejected_at_every_string_boundary(model, model_paths):
    for call in (lambda: clap_cpp.CLAP(str(model_paths[0]) + chr(0)),
                 lambda: model.tokenize("hello" + chr(0)),
                 lambda: model.encode_text("hello" + chr(0)),
                 lambda: model.encode_audio("missing.wav" + chr(0))):
        with pytest.raises(ValueError, match="NUL"):
            call()


def test_keyword_overloads_and_duplicate_arguments(model):
    tokens = model.tokenize("hello")
    assert model.encode_text(tokens=tokens) == model.encode_text(text="hello")
    for call in (lambda: model.encode_text("hello", tokens=tokens),
                 lambda: model.encode_text(tokens="hello"),
                 lambda: model.encode_audio(pcm="x.wav"),
                 lambda: model.encode_audio([0.1], pcm=[0.1]),
                 lambda: model.encode_audio([0.1], 24000, sample_rate=24000),
                 lambda: model.encode_audio(path="x.wav", sample_rate=24000),
                 lambda: model.encode_audio(path=123),
                 lambda: model.encode_audio(path="x.wav", pcm=[0.1])):
        with pytest.raises(TypeError):
            call()


def test_enter_closed_model_is_rejected(model):
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.__enter__()


def test_close_races_all_native_operations_safely(model_paths):
    from threading import Barrier
    instance = clap_cpp.CLAP(model_paths[0], threads=1)
    barrier = Barrier(5)
    operations = [lambda: instance.encode_text("hello"),
                  lambda: instance.tokenize("hello"),
                  lambda: instance.audio_sample_rate,
                  lambda: instance.projection_dimension,
                  instance.close]
    def run(operation):
        barrier.wait(timeout=10)
        try:
            return operation()
        except RuntimeError as exc:
            assert "closed" in str(exc)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run, operation) for operation in operations]
        for future in futures:
            future.result(timeout=30)
    instance.close()
    with pytest.raises(RuntimeError, match="closed"):
        instance.encode_text("hello")


def test_distinct_threads_copy_their_own_native_diagnostic(tmp_path):
    from threading import Barrier
    barrier = Barrier(4)
    def fail(index):
        filename = tmp_path / f"missing-thread-{index}.gguf"
        barrier.wait(timeout=10)
        with pytest.raises(RuntimeError) as error:
            clap_cpp.CLAP(filename, threads=1)
        assert filename.name in str(error.value)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fail, range(4)))


@pytest.mark.parametrize("frames", [False, True])
def test_conversion_failure_still_frees_native_allocation(model, mert, pcm, monkeypatch, frames):
    import importlib
    wrapper = importlib.import_module(clap_cpp.CLAP.__module__)
    name = "clap_free_audio_frames" if frames else "clap_free"
    original = getattr(wrapper._lib, name)
    freed = []

    def release(allocation):
        freed.append(True)
        original(allocation)

    def cannot_copy(*args):
        raise MemoryError("simulated Python result allocation failure")

    monkeypatch.setattr(wrapper._lib, name, release)
    monkeypatch.setattr(wrapper, "list", cannot_copy, raising=False)
    with pytest.raises(MemoryError, match="Python result allocation"):
        if frames:
            mert.audio_frames(pcm, sample_rate=24000)
        else:
            model.encode_text("hello")
    assert freed == [True]


def test_explicit_close_and_finalizer_destroy_exactly_once(model_paths, monkeypatch):
    import importlib
    wrapper = importlib.import_module(clap_cpp.CLAP.__module__)
    original = wrapper._lib.clap_destroy
    destroyed = []

    def destroy(handle):
        destroyed.append(handle.value)
        original(handle)

    monkeypatch.setattr(wrapper._lib, "clap_destroy", destroy)
    instance = clap_cpp.CLAP(model_paths[0], threads=1)
    instance.close()
    instance.close()
    instance.__del__()
    del instance
    gc.collect()
    assert len(destroyed) == 1


def test_model_lock_is_a_real_blocking_lock(model):
    import threading
    assert type(model._lock) is type(threading.Lock())
    assert model._lock.acquire(blocking=False)
    try:
        assert not model._lock.acquire(blocking=False)
    finally:
        model._lock.release()
