#!/usr/bin/env python3
"""Shared architecture contract execution; no model/converter dependencies.

Adapters describe real native commands and expected outputs, not subprocess
wrappers. Numerical oracle executables keep calling the public C API directly.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import subprocess
from typing import Callable, Optional, Sequence, Union
import wave


@dataclass
class Vector:
    dimension: int
    expected: Optional[Sequence[float]] = None
    tolerance: float = 0.0
    inclusive: bool = False

    def __call__(self, actual):
        assert isinstance(actual, list), f"expected vector, got {type(actual).__name__}"
        assert len(actual) == self.dimension, (len(actual), self.dimension)
        if self.expected is not None:
            assert len(self.expected) == self.dimension, "reference vector dimension mismatch"
            error = max(abs(a - b) for a, b in zip(actual, self.expected))
            assert (error <= self.tolerance if self.inclusive else error < self.tolerance), (
                f"max error {error}, tolerance {'<=' if self.inclusive else '<'} {self.tolerance}")


@dataclass
class Frames:
    frames: int
    dimension: int
    expected: Sequence[float]
    tolerance: float

    def __call__(self, actual):
        assert actual["frames"] == self.frames, (actual["frames"], self.frames)
        assert actual["dimension"] == self.dimension, (actual["dimension"], self.dimension)
        Vector(self.frames * self.dimension, self.expected, self.tolerance, inclusive=True)(actual["values"])


@dataclass
class Diagnostic:
    text: str
    ignore_case: bool = False

    def matches(self, stderr):
        return self.text in (stderr.lower() if self.ignore_case else stderr)


@dataclass
class Command:
    name: str
    args: Sequence[object]
    # None is reserved for paired contracts where both outcomes are valid.
    success: Optional[bool] = True
    diagnostics: Sequence[Union[str, Diagnostic]] = ()
    # Some pre-existing loader contracts accept any of several diagnostics.
    diagnostic_any: bool = False
    diagnostic_lower: bool = False
    positive_exit: bool = False
    nonempty_stderr: bool = False
    output: Optional[Callable] = None
    timeout: Optional[float] = None

    def run(self):
        argv = [str(arg) for arg in self.args]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            raise AssertionError(f"command timed out after {self.timeout}s: {argv!r}\n"
                                 f"stdout: {error.stdout!r}\nstderr: {error.stderr!r}") from error
        detail = (f"command: {argv!r}\nexit: {result.returncode}\n"
                  f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        try:
            if self.success is True:
                assert result.returncode == 0, "expected success"
            elif self.success is False:
                if self.positive_exit:
                    assert result.returncode > 0, "expected positive rejection exit (not a signal)"
                else:
                    assert result.returncode != 0, "expected rejection"
            if self.nonempty_stderr:
                assert result.stderr, "expected stderr diagnostic"
            if self.diagnostics:
                text = result.stderr.lower() if self.diagnostic_lower else result.stderr
                matches = [word.matches(result.stderr) if isinstance(word, Diagnostic) else word in text
                           for word in self.diagnostics]
                assert (any(matches) if self.diagnostic_any else all(matches)), (
                    f"expected {'any' if self.diagnostic_any else 'all'} diagnostics: {self.diagnostics!r}")
            if self.output is not None:
                self.output(json.loads(result.stdout))
        except (AssertionError, ValueError, KeyError, TypeError) as error:
            raise AssertionError(f"{error}\n{detail}") from error
        return result


@dataclass
class Compare:
    name: str
    commands: Sequence[Command]
    check: Callable

    def run(self):
        results = [command.run() for command in self.commands]
        try:
            self.check(results)
        except Exception as error:
            raise AssertionError(f"{error}\npaired command results: {results!r}") from error


@dataclass
class Check:
    """Architecture-specific converter/paired-output checks, never a C API facade."""
    name: str
    check: Callable

    def run(self):
        self.check()


@dataclass
class Context:
    cli: Optional[Path] = None
    build: Optional[Path] = None
    fixture: Optional[Path] = None
    work: Optional[Path] = None

    def embedding(self, model, *args):
        return [self.cli, "embedding", "--model", model, *args]


def run_cases(architecture, suite, cases, case_filter=None):
    selected = 0
    for case in cases:
        if case_filter is not None and case_filter not in case.name:
            continue
        selected += 1
        label = f"{architecture}/{suite}/{case.name}"
        try:
            case.run()
        except Exception as error:
            raise AssertionError(f"{label}: {error}") from error
        print(f"PASS {label}", flush=True)
    if not selected:
        raise ValueError(f"no cases matched {architecture}/{suite}: {case_filter!r}")
    print(f"{architecture}/{suite}: {selected} cases passed", flush=True)
    return selected


def pcm_silence(path, rate, samples):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\0\0" * samples)


def float_wave(path, rate, waveform):
    samples = struct.pack("<" + "f" * len(waveform), *waveform)
    path.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt " +
                     struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32) +
                     b"data" + struct.pack("<I", len(samples)) + samples)


def encode_string(text):
    value = text.encode()
    return struct.pack("<Q", len(value)) + value


def metadata_model(path, metadata, tensors=0, infos=b"", payload=b""):
    """Write string-metadata GGUF v3, retaining tensor offsets/payload verbatim."""
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, tensors, len(metadata)))
    for key, value in metadata.items():
        header += encode_string(key) + struct.pack("<I", 8) + encode_string(value)
    header += infos
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + payload)


def read_metadata_model(path):
    """Read the string-only synthetic fixtures, not arbitrary production GGUFs."""
    source = path.read_bytes()
    magic, version, tensors, count = struct.unpack_from("<4sIQQ", source)
    assert magic == b"GGUF" and version == 3, "expected GGUF v3 fixture"
    position = 24

    def read_string():
        nonlocal position
        size, = struct.unpack_from("<Q", source, position)
        position += 8
        value = source[position:position + size].decode()
        position += size
        return value

    metadata = {}
    for _ in range(count):
        key = read_string()
        kind, = struct.unpack_from("<I", source, position)
        position += 4
        assert kind == 8, f"fixture metadata {key} is not a string"
        metadata[key] = read_string()
    infos_start = position
    for _ in range(tensors):
        read_string()
        dimensions, = struct.unpack_from("<I", source, position)
        position += 4 + dimensions * 8 + 12
    return metadata, tensors, source[infos_start:position], source[(position + 31) // 32 * 32:]
