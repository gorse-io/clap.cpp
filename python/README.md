# clap.cpp Python bindings

## Installation

```sh
pip install clap-cpp           # CPU
pip install "clap-cpp[cuda]"   # NVIDIA CUDA
pip install "clap-cpp[rocm]"   # AMD ROCm
pip install "clap-cpp[vulkan]" # Vulkan
```

The extras install version-matched `clap-cpp-cuda`, `clap-cpp-rocm` or
`clap-cpp-vulkan` wheels. They do not change compiler flags during pip install.
The public import remains `from clap_cpp import Model`. With one GPU backend
installed it is selected automatically; otherwise the CPU extension is used.
Set `CLAP_CPP_BACKEND=cpu|cuda|rocm|vulkan` before importing to select explicitly.
If multiple GPU packages are installed, explicit selection is required.
`clap_cpp.backend` reports the selected compiled backend, not device availability.
A GPU extension that cannot load raises an error rather than silently loading
CPU; `CLAP_CPP_BACKEND=cpu` explicitly uses the bundled CPU extension.

Release wheels cover CPython 3.9–3.14 (not free-threaded builds):

| Backend | Platforms | Required system runtime |
| --- | --- | --- |
| CPU | Linux x86_64/ARM64 (glibc 2.35+), macOS ARM64, Windows x86_64 | None beyond platform runtimes |
| CUDA | Linux x86_64 (glibc 2.35+) | CUDA 12.9-compatible NVIDIA driver; cudart/cuBLAS are statically linked |
| ROCm | Linux x86_64 (glibc 2.35+) | ROCm 6.4.3 HIP, hipBLAS, rocBLAS (including library data), compatible AMD driver |
| Vulkan | Linux x86_64 (glibc 2.39+) | System Vulkan loader and vendor ICD/driver |

GPU wheels do not bundle drivers. ROCm libraries must be discoverable through
`ldconfig` or `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64`; keep rocBLAS's data
installed in the matching ROCm prefix. No Windows GPU or macOS GPU wheels are
built by this workflow. On other targets pip can build from a source archive,
but this requires the corresponding GPU SDK and compiler.

GPU acceleration currently uses the native runtime's cached dense matrix
multiplication path for CLAP. Architectures requiring storage-only precision
(MERT, MuQ, MuQ-MuLan and CLaMP3) and other operations retain their CPU paths.
Device initialization failures and unsupported operations retain the native
CPU fallback; installing an extra does not guarantee every operation runs on GPU.

The commands above require these projects to have been published to PyPI.
Before the first publication, download matching wheels from a GitHub Release
into a local directory and use, for example,
`pip install --find-links ./wheels "clap-cpp[cuda]"`.

## Build from source

Build and install from a recursive checkout:

```sh
git clone --recursive https://github.com/gorse-io/clap.cpp.git
cd clap.cpp
python -m pip install ./python
```

Requires Python 3.9+, a C++17 compiler and CMake 3.20+. The build backend installs
CMake/Ninja if necessary. For the default CPU package NumPy is the only runtime
Python dependency; PyTorch, FFmpeg, a separate `clap` executable and GPU drivers
are not required. The extension statically links the CPU runtime and bundled
ICU. The base package disables GPU backends and OpenMP. x86 wheels use baseline
instructions rather than the build machine's instruction set. No pretrained
weights are downloaded automatically.

Build an optional backend project separately, with its GPU SDK installed:

```sh
python python/prepare_backend.py cuda ./build-python-cuda
python -m pip install ./build-python-cuda
python -m pip install ./python
```

Replace `cuda` with `rocm` or `vulkan` as needed. The helper also works inside
the unpacked base source distribution. Extra CMake options can be passed with
pip's `-Ccmake.define.NAME=VALUE`, for example CUDA/HIP architecture lists when
building on a machine without a GPU. GPU backend selection and wheel package
names must match: do not enable a GPU backend directly on the base project.

## Usage

Convert a supported checkpoint to GGUF with the repository's conversion tools,
then load it once and reuse it:

```python
from clap_cpp import Model, decode_audio

with Model("model.gguf", threads=4) as model:
    text_embedding = model.text("A dog barking")
    pcm, sample_rate = decode_audio("dog.wav")
    audio_embedding = model.audio(pcm, sample_rate=sample_rate)
    print(model.projection_dimension, model.audio_sample_rate)
```

`Model` uses the same GGUF architecture dispatch as the native library (CLAP,
CLaMP3, MERT-v1, MERT-v2, MuQ and MuQ-MuLan). An operation unsupported by a
particular architecture raises a native error; for example audio-only encoders
cannot embed text. CLaMP3 requires a GGUF with its bundled MERT frontend.

## API

- `Model(path, threads=0)`: `str`/`pathlib.Path`; zero selects the native thread
  default. `close()` is idempotent and the context manager closes on exit.
- `model.text(text_or_tokens)`: one UTF-8 string or token-ID sequence to a
  one-dimensional float32 NumPy embedding.
- `model.tokenize(text, max_length=512)`: native tokenizer returning a list of
  integer IDs. Its limit must be compatible with the loaded model; strings may
  be truncated by the tokenizer. Prefer `text(str)` for normal use.
- `model.audio(pcm, sample_rate=48000, crop_offset=0)`: mono waveform to a float32
  embedding. Automatic model-rate resampling; crop offset uses the native
  model-rate sample units and is not supported by every architecture.
- `model.audio_frames(pcm, sample_rate=48000)`: native frame representations as a
  `[frames, dimension]` float32 array, for architectures exposing frames.
- `model.clamp3_features(pcm, sample_rate=24000)`: MERT-v1 CLaMP3 preprocessing
  features as a `[windows, dimension]` float32 array (at least one second).
- `model.projection_dimension`, `model.audio_sample_rate`: read-only integers.
- `decode_audio(path)`: `(pcm, sample_rate)`; decodes WAV, MP3 or FLAC, averages
  channels to mono, preserves sample rate.
- `resample_audio(pcm, source_rate, target_rate=48000)`: mono float32 array using
  the native resampler.

PCM accepts one-dimensional numeric arrays or sequences. Noncontiguous arrays
and other numeric dtypes are converted to contiguous float32; the binding copies
inputs before releasing the GIL, so inference never retains a Python buffer.
Empty, nonfinite or multidimensional PCM and nonpositive sample rates are
rejected. Returned arrays own their storage and survive model closure.
Native invalid arguments map to `ValueError`, other native runtime errors to
`RuntimeError`. Calls after closing raise `RuntimeError`.

The binding releases the GIL for inference/loading/decoding/resampling and
serializes all operations on each model (including closing). Different model
instances can execute concurrently. Embedding normalization is architecture
specific and identical to the C++ API; the binding does not normalize outputs.

## Development and source distributions

```sh
python -m pip install './python[test]' build
python -m pytest python/tests
python -m build python
```

The source distribution includes the pinned native sources and third-party
licenses; a wheel can be rebuilt from that archive without a Git checkout or
CMake-time downloads. Build dependencies must still be available from the Python
package index or a local wheelhouse.

## Releases

`.github/workflows/python-release.yml` runs when a GitHub Release is published.
It builds CPU, CUDA, ROCm and Vulkan distributions without running tests, using
sccache for C, C++, CUDA and HIP compilation. Cache storage reuses the repository's
`SCCACHE_BUCKET`, `SCCACHE_ENDPOINT`, `SCCACHE_REGION` variables and
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` secrets. Linux wheels are repaired
with auditwheel; system GPU libraries remain external as listed above. All
wheels and source distributions are attached to that release.

Stable releases also publish the four distributions to PyPI using OIDC. Before
releasing, configure a PyPI trusted publisher (owner `gorse-io`, repository
`clap.cpp`, workflow `python-release.yml`, environment `pypi`) for **each** of
`clap-cpp`, `clap-cpp-cuda`, `clap-cpp-rocm` and `clap-cpp-vulkan`, and configure
any required GitHub `pypi` environment approvals. Prereleases only attach assets.
The version in `python/pyproject.toml` and its three exact extra dependencies
must be updated together before publishing a new version. The backend helper
copies that version into each backend distribution; existing PyPI versions
cannot be overwritten.
