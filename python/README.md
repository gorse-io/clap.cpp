# clap.cpp Python bindings

## Install

```sh
pip install clap-cpp           # CPU
pip install "clap-cpp[cuda]"   # NVIDIA CUDA
pip install "clap-cpp[rocm]"   # AMD ROCm
pip install "clap-cpp[vulkan]" # Vulkan
```

The extras install version-matched `clap-cpp-cuda`, `clap-cpp-rocm` or
`clap-cpp-vulkan` wheels. They do not change compiler flags during pip install.
The public import remains `from clap_cpp import CLAP`. With one GPU backend
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

## Type hints

The package includes type stubs and a `py.typed` marker for IDE completion and
static checking with mypy or Pyright. `encode_text` and `encode_audio` return
`list[float]`; `audio_frames` and `clamp3_features` return time-major
`list[list[float]]`. PCM input accepts mono numeric sequences such as lists or
tuples. The runtime has no NumPy dependency. Text, token-sequence, audio-file and
PCM input overloads are declared separately.

## Usage

```python
from clap_cpp import CLAP

with CLAP("model.gguf", threads=4) as model:
    text_embedding = model.encode_text("A dog barking")
    audio_embedding = model.encode_audio("dog.wav")
    print(model.projection_dimension, model.audio_sample_rate)
```

## Build from source

Build and install from a recursive checkout:

```sh
git clone --recursive https://github.com/gorse-io/clap.cpp.git
cd clap.cpp
python -m pip install ./python
```

Requires Python 3.9+, a C++17 compiler and CMake 3.20+. The build backend installs
CMake/Ninja if necessary. The CPU package has no runtime Python dependencies;
PyTorch, FFmpeg, a separate `clap` executable and GPU drivers are not required.
The extension statically links the CPU runtime and bundled ICU. The base package
disables GPU backends and OpenMP. x86 wheels use baseline instructions rather
than the build machine's instruction set. No pretrained weights are downloaded
automatically.

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
