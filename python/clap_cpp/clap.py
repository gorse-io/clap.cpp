"""NumPy-free bindings to the package-local clap.cpp public C ABI."""
from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
import math
import operator
import os
from pathlib import Path
import sys
import threading


class _Options(C.Structure):
    _fields_ = [("threads", C.c_int), ("device", C.c_char_p),
                ("activation_precision", C.c_int)]


class _Device(C.Structure):
    _fields_ = [("name", C.c_char_p), ("description", C.c_char_p), ("gpu", C.c_bool)]


class _Frames(C.Structure):
    _fields_ = [("values", C.POINTER(C.c_float)), ("frames", C.c_size_t),
                ("dimension", C.c_size_t)]


# Never consult find_library, PATH, or another installed backend. Each package
# owns its native payload. CDLL (not PyDLL) releases the GIL around foreign calls.
_directory = Path(__file__).resolve().parent
_filename = "clap.dll" if sys.platform == "win32" else (
    "libclap.dylib" if sys.platform == "darwin" else "libclap.so")
_dll_directory = None
try:
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        _dll_directory = os.add_dll_directory(str(_directory))
    _lib = C.CDLL(str(_directory / _filename))
except OSError as exc:
    raise ImportError(f"Cannot load package-local clap.cpp native library {_directory / _filename}: {exc}") from exc


def _bind(name, result, *arguments):
    function = getattr(_lib, name)
    function.argtypes = list(arguments)
    function.restype = result
    return function


_Handle = C.c_void_p
_Ints = C.POINTER(C.c_int)
_Floats = C.POINTER(C.c_float)
_Size = C.c_size_t
_bind("clap_default_options", _Options)
_bind("clap_last_error", C.c_char_p)
_bind("clap_create", C.c_int, C.c_char_p, C.POINTER(_Options), C.POINTER(_Handle))
_bind("clap_destroy", None, _Handle)
_bind("clap_available_devices", C.c_int, C.POINTER(C.POINTER(_Device)), C.POINTER(_Size))
_bind("clap_free_devices", None, C.POINTER(_Device), _Size)
_bind("clap_free", None, C.c_void_p)
_bind("clap_free_audio_frames", None, C.POINTER(_Frames))
_bind("clap_tokenize", C.c_int, _Handle, C.c_char_p, _Size, C.POINTER(_Ints), C.POINTER(_Size))
_bind("clap_encode_text", C.c_int, _Handle, C.c_char_p, C.POINTER(_Floats), C.POINTER(_Size))
_bind("clap_encode_tokens", C.c_int, _Handle, _Ints, _Size, C.POINTER(_Floats), C.POINTER(_Size))
_bind("clap_encode_audio", C.c_int, _Handle, _Floats, _Size, C.c_int, _Size, C.POINTER(_Floats), C.POINTER(_Size))
_bind("clap_encode_audio_file", C.c_int, _Handle, C.c_char_p, _Size, C.POINTER(_Floats), C.POINTER(_Size))
_bind("clap_get_audio_frames", C.c_int, _Handle, _Floats, _Size, C.c_int, C.POINTER(_Frames))
_bind("clap_audio_sample_rate", C.c_int, _Handle)
_bind("clap_projection_dimension", C.c_int, _Handle)


def _check(status):
    if status:
        # c_char_p copies the thread-local diagnostic into Python bytes NOW,
        # before frees, another fallible call, or releasing the model lease.
        raw = _lib.clap_last_error()
        message = raw.decode("utf-8", errors="replace") if raw else "clap.cpp native call failed"
        raise {1: ValueError, 2: MemoryError}.get(status, RuntimeError)(message)


def _integer(value, name, minimum=0, maximum=(1 << (C.sizeof(C.c_int) * 8 - 1)) - 1):
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if result < minimum:
        if minimum < 0:
            raise OverflowError(f"{name} exceeds its native integer range")
        raise ValueError(f"{name} must be {'positive' if minimum == 1 else 'nonnegative'}")
    if result > maximum:
        raise OverflowError(f"{name} exceeds its native integer range")
    return result


_SIZE_MAX = (1 << (C.sizeof(_Size) * 8)) - 1
_INT_MIN = -(1 << (C.sizeof(C.c_int) * 8 - 1))
_MISSING = object()


def _string(value):
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    if "\0" in value:
        raise ValueError("text and paths must contain no NUL bytes")
    return value.encode("utf-8")


def _path(value):
    value = os.fspath(value)
    if not isinstance(value, str):
        raise TypeError("path must be str or PathLike[str]")
    return _string(value)


def _sequence(value):
    if isinstance(value, (str, bytes, bytearray, dict)) or not (
            hasattr(value, "__len__") and hasattr(value, "__getitem__")):
        raise TypeError("expected a numeric sequence")
    return value


def _pcm(value):
    value = _sequence(value)
    if not len(value):
        raise ValueError("PCM must be a nonempty mono sequence")
    samples = []
    for sample in value:
        if isinstance(sample, (str, bytes, bytearray)):
            raise TypeError("PCM samples must be numeric")
        try:
            sample = float(sample)
        except OverflowError:
            raise ValueError("PCM samples must be finite F32 values") from None
        # Nested/non-numeric elements deliberately retain float()'s TypeError.
        if not math.isfinite(sample) or not math.isfinite(C.c_float(sample).value):
            raise ValueError("PCM samples must be finite F32 values")
        samples.append(sample)
    return (C.c_float * len(samples))(*samples)


class CLAP:
    """Own a shared native model with independent concurrent inference calls.

    close() rejects new calls immediately. If calls are in flight, the last
    caller destroys the model after copying and freeing its native results.
    User input conversion runs before acquiring a lifecycle lease.
    """

    def __init__(self, path, threads=0):
        if hasattr(self, "_lock"):
            raise RuntimeError("CLAP is already initialized")
        self._lock = threading.Lock()
        self._handle = _Handle()
        self._closed = False
        self._active = 0
        # Keep the CDLL alive even during interpreter shutdown/finalization.
        self._native = _lib
        filename = _path(path)
        threads = _integer(threads, "threads")
        with self._lock:
            if self._closed:
                raise RuntimeError("CLAP is closed")
            options = self._native.clap_default_options()
            options.threads = threads
            _check(self._native.clap_create(filename, C.byref(options), C.byref(self._handle)))

    def __copy__(self):
        raise TypeError("CLAP owns a native model and cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("CLAP owns a native model and cannot be copied")

    def _ensure_open(self):
        if self._closed or not self._handle:
            raise RuntimeError("CLAP is closed")

    @contextmanager
    def _lease(self):
        # Only pin/unpin under the lock: native calls release the GIL and must
        # not hold the lifecycle lock or serialize other inference calls.
        with self._lock:
            self._ensure_open()
            handle = self._handle
            self._active += 1
        try:
            yield handle
        finally:
            destroy = None
            with self._lock:
                self._active -= 1
                if self._closed and not self._active:
                    destroy, self._handle = self._handle, _Handle()
            if destroy:
                self._native.clap_destroy(destroy)

    def _buffer(self, function, *args, integer=False):
        with self._lease() as handle:
            values = _Ints() if integer else _Floats()
            count = _Size()
            try:
                _check(function(handle, *args, C.byref(values), C.byref(count)))
                return list(values[:count.value]) if count.value else []
            finally:
                self._native.clap_free(values)

    def _property(self, function):
        with self._lease() as handle:
            return function(handle)

    @property
    def projection_dimension(self):
        return self._property(self._native.clap_projection_dimension)

    @property
    def audio_sample_rate(self):
        return self._property(self._native.clap_audio_sample_rate)

    def tokenize(self, text, max_length=512):
        text = _string(text)
        length = _integer(max_length, "max_length", 1, _SIZE_MAX)
        return self._buffer(self._native.clap_tokenize, text, length, integer=True)

    def encode_text(self, text=_MISSING, *, tokens=_MISSING):
        tokens_keyword = tokens is not _MISSING
        if text is _MISSING:
            if tokens is _MISSING:
                raise TypeError("encode_text requires text or tokens")
            text = tokens
        elif tokens is not _MISSING:
            raise TypeError("specify either text or tokens, not both")
        if isinstance(text, str) and not tokens_keyword:
            return self._buffer(self._native.clap_encode_text, _string(text))
        items = [_integer(token, "token", _INT_MIN) for token in _sequence(text)]
        values = (C.c_int * len(items))(*items)
        return self._buffer(self._native.clap_encode_tokens, values, len(items))

    def encode_audio(self, *args, **kwargs):
        # Preserve both pybind overloads, including keyword names and the file
        # overload's second positional crop_offset (not PCM sample_rate).
        if len(args) > 3:
            raise TypeError("encode_audio accepts at most three positional arguments")
        input_kind = None
        if args:
            value = args[0]
            if "path" in kwargs or "pcm" in kwargs:
                raise TypeError("multiple values for audio input")
        else:
            keys = [key for key in ("path", "pcm") if key in kwargs]
            if len(keys) != 1:
                raise TypeError("encode_audio requires exactly one path or pcm")
            input_kind = keys[0]
            value = kwargs.pop(input_kind)
            if input_kind == "path" and not isinstance(value, (str, os.PathLike)):
                raise TypeError("path must be str or PathLike[str]")
        is_path = input_kind != "pcm" and isinstance(value, (str, os.PathLike))
        names = ("crop_offset",) if is_path else ("sample_rate", "crop_offset")
        if len(args) - 1 > len(names):
            raise TypeError("too many arguments for file audio")
        for name, value_arg in zip(names, args[1:]):
            if name in kwargs:
                raise TypeError(f"multiple values for {name}")
            kwargs[name] = value_arg
        if any(key not in names for key in kwargs):
            raise TypeError("unexpected encode_audio keyword argument")
        crop = _integer(kwargs.get("crop_offset", 0), "crop_offset", 0, _SIZE_MAX)
        if is_path:
            return self._buffer(self._native.clap_encode_audio_file, _path(value), crop)
        rate = _integer(kwargs.get("sample_rate", 48000), "sample_rate", 1)
        samples = _pcm(value)
        return self._buffer(self._native.clap_encode_audio, samples, len(samples), rate, crop)

    def audio_frames(self, pcm, sample_rate=48000):
        rate = _integer(sample_rate, "sample_rate", 1)
        samples = _pcm(pcm)
        with self._lease() as handle:
            frames = _Frames()
            try:
                _check(self._native.clap_get_audio_frames(
                    handle, samples, len(samples), rate, C.byref(frames)))
                width = frames.dimension
                return [list(frames.values[i * width:(i + 1) * width]) if width else []
                        for i in range(frames.frames)]
            finally:
                self._native.clap_free_audio_frames(C.byref(frames))

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        destroy = None
        with lock:
            self._closed = True
            if not self._active:
                destroy, self._handle = self._handle, _Handle()
        # Defer rather than wait when leased. This also lets user protocols
        # call close() reentrantly without waiting for another caller.
        if destroy:
            self._native.clap_destroy(destroy)

    def __enter__(self):
        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Never leak an unraisable exception from partial initialization or
            # interpreter shutdown. Explicit close still reports failures.
            pass
