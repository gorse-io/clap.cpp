"""CPU audio/text embeddings backed by the native clap.cpp runtime."""

from ._clap_cpp import Model, decode_audio, resample_audio

__all__ = ["Model", "decode_audio", "resample_audio"]
