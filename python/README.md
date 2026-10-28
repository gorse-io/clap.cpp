# clap.cpp Python bindings

[![PyPI](https://img.shields.io/pypi/v/clap-cpp)](https://pypi.org/project/clap-cpp/)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/clap-cpp)
[![PyPI downloads](https://img.shields.io/pypi/dm/clap-cpp)](https://pypistats.org/packages/clap-cpp)
![PyPI Format](https://img.shields.io/pypi/format/clap-cpp)

## Install

```sh
pip install clap-cpp           # CPU (Metal on macOS)
pip install "clap-cpp[cuda]"   # NVIDIA CUDA
pip install "clap-cpp[rocm]"   # AMD ROCm
pip install "clap-cpp[vulkan]" # Vulkan
```

| Backend | Platforms | Required system runtime |
| --- | --- | --- |
| CPU (Metal on macOS) | Linux x86_64/ARM64, macOS ARM64, Windows x86_64 | |
| CUDA | Linux x86_64 (glibc 2.35+) | CUDA 12.9-compatible NVIDIA driver; cudart/cuBLAS are statically linked |
| ROCm | Linux x86_64 (glibc 2.35+) | ROCm 6.4.3 HIP, hipBLAS, rocBLAS (including library data), compatible AMD driver |
| Vulkan | Linux x86_64 (glibc 2.39+) | System Vulkan loader and vendor ICD/driver |

## Usage

```python
from clap_cpp import CLAP

with CLAP("larger_clap_general-q4_k_m.gguf") as model:
    text_embedding = model.encode_text("Sound of a dog")
    audio_embedding = model.encode_audio("dog.wav")
```

For EmbeddingGemma 2 audio, also load the matching audio mmproj GGUF:

```python
with CLAP("embeddinggemma2.gguf", threads=4, mmproj="mmproj-Q8_0.gguf") as model:
    audio_embedding = model.encode_audio("recording.wav")
```

`mmproj` accepts a string or `pathlib.Path`. Text still works without it; the
existing positional `threads` argument is unchanged. Missing/incompatible
projectors raise native errors. Image encoding is not supported.
