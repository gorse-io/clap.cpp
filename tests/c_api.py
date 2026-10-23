"""Real exported-symbol/inference coverage; standard-library ctypes only."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import wave
from c_api_fixture import clap_fixture, mert_fixture

class ModelParams(C.Structure):
    _fields_ = [('n_threads', C.c_int32)]
class AudioParams(C.Structure):
    _fields_ = [('sample_rate', C.c_int32), ('crop_offset', C.c_size_t)]
class Floats(C.Structure):
    _fields_ = [('data', C.POINTER(C.c_float)), ('size', C.c_size_t)]
class Tokens(C.Structure):
    _fields_ = [('data', C.POINTER(C.c_int32)), ('size', C.c_size_t)]
class Audio(C.Structure):
    _fields_ = [('samples', C.POINTER(C.c_float)), ('n_samples', C.c_size_t), ('sample_rate', C.c_int32)]
class Frames(C.Structure):
    _fields_ = [('values', C.POINTER(C.c_float)), ('n_values', C.c_size_t),
                ('frames', C.c_int32), ('dimension', C.c_int32)]

lib = C.CDLL(sys.argv[1])
def bind(name, result, *args):
    f = getattr(lib, name)
    f.restype, f.argtypes = result, list(args)
    return f
P = C.POINTER
handle = C.c_void_p
last_error = bind('clap_last_error', C.c_char_p)
model_default = bind('clap_model_default_params', ModelParams)
audio_default = bind('clap_audio_default_params', AudioParams)
load = bind('clap_model_load', C.c_int, C.c_char_p, ModelParams, P(handle))
free = bind('clap_model_free', None, handle)
text = bind('clap_model_text', C.c_int, handle, C.c_char_p, P(Floats))
tokenize = bind('clap_model_tokenize', C.c_int, handle, C.c_char_p, C.c_size_t, P(Tokens))
text_tokens = bind('clap_model_text_tokens', C.c_int, handle, P(C.c_int32), C.c_size_t, P(Floats))
audio = bind('clap_model_audio', C.c_int, handle, P(C.c_float), C.c_size_t, AudioParams, P(Floats))
frames = bind('clap_model_audio_frames', C.c_int, handle, P(C.c_float), C.c_size_t, C.c_int32, P(Frames))
features = bind('clap_model_clamp3_features', C.c_int, handle, P(C.c_float), C.c_size_t, C.c_int32, P(Frames))
dimension = bind('clap_model_projection_dimension', C.c_int, handle, P(C.c_int32))
sample_rate = bind('clap_model_audio_sample_rate', C.c_int, handle, P(C.c_int32))
release = bind('clap_float_buffer_free', None, P(Floats))
release_tokens = bind('clap_token_buffer_free', None, P(Tokens))
release_frames = bind('clap_audio_frames_free', None, P(Frames))
decode = bind('clap_audio_decode', C.c_int, C.c_char_p, P(Audio))
release_audio = bind('clap_audio_free', None, P(Audio))

def success(code):
    assert code == 0, (code, last_error())
def vector(function, *args):
    result = Floats()
    success(function(*args, C.byref(result)))
    try:
        values = list(result.data[:result.size])
        assert values and all(math.isfinite(x) for x in values)
        return values
    finally:
        release(C.byref(result))
def near(a, b, eps=1e-5):
    assert len(a) == len(b), (len(a), len(b))
    assert max(abs(x-y) for x, y in zip(a, b)) < eps, (a[:8], b[:8])
def cli(path, *args):
    p = subprocess.run([sys.argv[2], '--model', str(path), *args], capture_output=True, text=True, check=True)
    return json.loads(p.stdout)

with tempfile.TemporaryDirectory(prefix='clap-c-api-') as temp:
    root = Path(temp)
    model_path, mert_path = root/'clap.gguf', root/'mert.gguf'
    clap_fixture(model_path)
    mert_fixture(mert_path)
    if len(sys.argv) > 3:
        subprocess.run([sys.argv[3], str(model_path)], check=True)
    m = handle()
    params = model_default()
    params.n_threads = 2
    success(load(str(model_path).encode(), params, C.byref(m)))
    try:
        for func, expected in [(dimension, 3), (sample_rate, 48000)]:
            n = C.c_int32()
            success(func(m, C.byref(n)))
            assert n.value == expected
        prompts = ['hello', 'world', '你好 café']
        expected = {s: cli(model_path, '--text', s) for s in prompts}
        for prompt in prompts:
            actual = vector(text, m, prompt.encode())
            near(actual, expected[prompt])
            assert abs(sum(x*x for x in actual)-1) < 1e-5
            tokens = Tokens()
            success(tokenize(m, prompt.encode(), 512, C.byref(tokens)))
            try:
                near(vector(text_tokens, m, tokens.data, tokens.size), actual)
            finally:
                release_tokens(C.byref(tokens))
        assert any(abs(a-b) > 1e-7 for a, b in zip(expected['hello'], expected['world']))
        # CDLL releases the GIL: these actually enter the same native handle concurrently.
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [(p, executor.submit(vector, text, m, p.encode())) for p in prompts*4]
            for p, f in futures:
                near(f.result(), expected[p])
        tokens = Tokens()
        assert tokenize(m, b'x', 0, C.byref(tokens)) == 1
        result = Floats()
        bad = (C.c_int32*1)(-1)
        assert text_tokens(m, bad, 1, C.byref(result)) != 0
        assert not result.data and result.size == 0
        assert text(m, None, C.byref(result)) == 1
        assert text(m, b'x', None) == 1
        assert audio(m, None, 1, audio_default(), C.byref(result)) == 1
        result_frames = Frames()
        pcm = (C.c_float*4800)(*(0.2*math.sin(i*0.13) for i in range(4800)))
        assert frames(m, pcm, len(pcm), 48000, C.byref(result_frames)) != 0
        assert not result_frames.values and result_frames.n_values == result_frames.frames == result_frames.dimension == 0
        assert features(m, pcm, len(pcm), 48000, C.byref(result_frames)) != 0
        # Error bytes and pointer survive operations on another thread.
        error = last_error()
        barrier = threading.Barrier(2)
        def other_thread():
            barrier.wait()
            near(vector(text, m, b'hello'), expected['hello'])
            assert last_error() == b''
            barrier.wait()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(other_thread)
            barrier.wait(); barrier.wait(); future.result()
        assert last_error() == error and error
        # Decode a real WAV and embed through the waveform-capable 64-mel model.
        wav_path = root/'input.wav'
        import struct
        with wave.open(str(wav_path), 'wb') as stream:
            stream.setparams((1, 2, 48000, len(pcm), 'NONE', 'not compressed'))
            stream.writeframes(struct.pack('<'+'h'*len(pcm), *(int(x*32767) for x in pcm)))
        decoded = Audio()
        success(decode(str(wav_path).encode(), C.byref(decoded)))
        try:
            ap = audio_default()
            actual = vector(audio, m, decoded.samples, decoded.n_samples, ap)
            near(actual, cli(model_path, '--audio', str(wav_path)))
            assert len(actual) == 3
        finally:
            release_audio(C.byref(decoded))
        # Independent resampled-input path at the model boundary.
        ap.sample_rate = 24000
        assert len(vector(audio, m, pcm, len(pcm), ap)) == 3
    finally:
        free(m)
    success(load(str(mert_path).encode(), params, C.byref(m)))
    try:
        n = 24000
        pcm = (C.c_float*n)(*(0.2*math.sin(i*0.13) for i in range(n)))
        mert_wav = root/'mert.wav'
        with wave.open(str(mert_wav), 'wb') as stream:
            stream.setparams((1, 4, 24000, n, 'NONE', 'not compressed'))
            stream.writeframes(struct.pack('<'+'i'*n, *(int(x*2147483647) for x in pcm)))
        # Match the file decoder's quantized PCM for CLI/native equivalence.
        decoded = Audio()
        success(decode(str(mert_wav).encode(), C.byref(decoded)))
        pcm = (C.c_float*n)(*decoded.samples[:decoded.n_samples])
        release_audio(C.byref(decoded))
        sequence = Frames()
        success(frames(m, pcm, n, 24000, C.byref(sequence)))
        try:
            assert sequence.frames == 75 and sequence.dimension == 4
            assert sequence.n_values == sequence.frames*sequence.dimension
            values = list(sequence.values[:sequence.n_values])
            expected_frames = cli(mert_path, '--audio', str(mert_wav), '--frames')
            assert expected_frames['frames'] == sequence.frames and expected_frames['dimension'] == sequence.dimension
            near(values, expected_frames['values'])
            pooled = [sum(values[i] for i in range(j, len(values), 4))/sequence.frames for j in range(4)]
            ap = audio_default(); ap.sample_rate = 24000
            near(vector(audio, m, pcm, n, ap), pooled)
            assert all(math.isfinite(x) for x in values)
        finally:
            release_frames(C.byref(sequence))
        success(features(m, pcm, n, 24000, C.byref(sequence)))
        assert sequence.frames == 1 and sequence.dimension == 4 and sequence.n_values == 4
        expected_features = cli(mert_path, '--audio', str(mert_wav), '--clamp3-features')
        near(list(sequence.values[:sequence.n_values]), expected_features['values'])
        assert expected_features['frames'] == 1 and expected_features['dimension'] == 4
        release_frames(C.byref(sequence))
        assert text(m, b'hello', C.byref(Floats())) != 0
    finally:
        free(m)
print('ctypes real text/token/audio/frame/features inference and concurrency passed')
