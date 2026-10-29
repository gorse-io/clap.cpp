"""CPU audio/text embeddings backed by the native clap.cpp runtime."""

from ._clap_cpp import CLAP as CLAP

__all__ = ["CLAP"]
