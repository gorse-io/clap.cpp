# clap.cpp Python bindings

[![PyPI](https://img.shields.io/pypi/v/clap-cpp)](https://pypi.org/project/clap-cpp/)
[![PyPI downloads](https://img.shields.io/pypi/dm/clap-cpp)](https://pypistats.org/packages/clap-cpp)

## Install

```sh
pip install clap-cpp           # Metal on macOS, CPU elsewhere
pip install "clap-cpp[cuda]"   # NVIDIA CUDA
pip install "clap-cpp[rocm]"   # AMD ROCm
pip install "clap-cpp[vulkan]" # Vulkan
```

The extras install version-matched `clap-cpp-cuda`, `clap-cpp-rocm` or
`clap-cpp-vulkan` wheels. They do not change compiler flags during pip install.
The public import remains `from clap_cpp import CLAP`. With one GPU backend
installed it is selected automatically; otherwise the base native library is used
(Metal on macOS, CPU elsewhere).
Set `CLAP_CPP_BACKEND=cpu|cuda|rocm|vulkan` before importing to select explicitly.
If multiple GPU packages are installed, explicit selection is required.
`clap_cpp.backend` reports the selected compiled backend, not device availability.
A GPU native library that cannot load raises an error rather than silently loading
the base runtime; `CLAP_CPP_BACKEND=cpu` explicitly uses the bundled base library.
For compatibility, its package selector and `clap_cpp.backend` value remain `cpu`
even on macOS. This selects a library, not the execution device: set
`CLAP_DEVICE=cpu` before loading a model to force CPU execution, or
`CLAP_METAL=0` to disable automatic Metal selection.

The wrapper uses standard-library `ctypes` to load the bundled C API library;
there is no pybind11 bridge or CPython extension. Platform wheels are tagged
`py3-none` and can be shared across supported Python versions (Python 3.9+).
Free-threaded interpreter builds are not currently part of release validation:

| Backend | Platforms | Required system runtime |
| --- | --- | --- |
| Base (Metal on macOS, CPU elsewhere) | Linux x86_64/ARM64 (glibc minimum recorded in wheel tags), macOS ARM64, Windows x86_64 | None beyond platform runtimes |
| CUDA | Linux x86_64 (glibc 2.35+) | CUDA 12.9-compatible NVIDIA driver; cudart/cuBLAS are statically linked |
| ROCm | Linux x86_64 (glibc 2.35+) | ROCm 6.4.3 HIP, hipBLAS, rocBLAS (including library data), compatible AMD driver |
| Vulkan | Linux x86_64 (glibc 2.39+) | System Vulkan loader and vendor ICD/driver |

GPU wheels do not bundle drivers. ROCm libraries must be discoverable through
`ldconfig` or `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64`; keep rocBLAS's data
installed in the matching ROCm prefix. No Windows GPU wheels or separate macOS
GPU wheels are built by this workflow; Metal is included in the base macOS wheel.
Source archives contain wrappers and packaging helpers only. They require an
externally built runtime to be staged before building a wheel; pip does not
compile a native library or install a GPU SDK.

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
`list[float]`; `audio_frames` returns time-major `list[list[float]]`.
CLaMP3 extracts MERT features internally through `encode_audio`; intermediate
feature extraction is not part of the public API.
PCM input accepts mono numeric sequences such as lists or
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

## Package an external runtime

Python packaging is a separate, package-only step. Build the native libraries
first using the repository-root `libraries` CMake preset (or download a matching
runtime artifact). Python never invokes CMake, Ninja or a compiler. The native
build is responsible for portable ISA flags, static dependencies, Metal/GPU
configuration and platform deployment targets.

For a runtime artifact with this layout:

```text
libraries/linux-x64/
  clap_cpp/
    libclap.so       # libclap.dylib on macOS, clap.dll on Windows
    licenses/...
    companions/...  # optional package-local binaries/data/notices
```

stage it into the wrapper project and build a platform wheel:

```sh
python -m pip install build 'scikit-build-core>=1.1.2'
cp -R libraries/linux-x64/clap_cpp/. python/clap_cpp/
cp python/LICENSE python/clap_cpp/licenses/clap-cpp.txt
python -m build --wheel python
python -m pip install python/dist/clap_cpp-*.whl
```

Copy only trusted, matching-platform runtime artifacts. Preserve all relative
paths and dependency notices beneath `clap_cpp/`; sibling library directories
need separate loader-aware wheel repair, not relocation into the package.
Start from a fresh wrapper project when switching runtimes so stale companion
libraries cannot leak into the next wheel. Do not copy Python sources from a
runtime artifact or overwrite the wrapper files.

The direct `scikit_build_core.build` backend only assembles the copied files:
`wheel.cmake=false`, `wheel.platlib=true`, `wheel.py-api="py3"` and
`sdist.cmake=false`. Wheel force-inclusion preserves Git-ignored runtime files;
explicit source inclusion keeps sdists wrapper-only. There is no custom backend,
build hook, staging or generation script.

Actions validates a nonempty host-platform library with the expected binary
header and a nonempty `licenses/clap-cpp.txt` before building wheels. The final
release gate validates the archived payload and notices again before publication.
Local builds must perform equivalent checks: the direct backend does **not**
reject an unstaged project, and a wheel without its matching native runtime is
unusable and must never be published. Metadata and wrapper-only source archives
need no runtime. Editable installs reference staged sources and are development
artifacts, not distributable runtime wheels. A local Linux wheel has a
`linux_<arch>` tag and still needs portability checks/repair before release.

On macOS, scikit-build-core selects the executing architecture via packaging's
platform tags, rather than treating a universal2 interpreter as proof of a
universal2 runtime. Keep `MACOSX_DEPLOYMENT_TARGET=11.0` for both the external
native build and wheel assembly to preserve the supported deployment minimum.
For an ARM64 runtime built at that minimum on an ARM64 host:

```sh
MACOSX_DEPLOYMENT_TARGET=11.0 python -m build --wheel python
```

This does not convert runtime architectures or change an existing binary's
minimum. Cross-packaging requires explicit matching platform configuration;
never label a single-architecture runtime universal2.

### Optional backend projects

`pyproject.toml.j2` is the shared CPU/CUDA/ROCm/Vulkan metadata template. The
checked-in `pyproject.toml` is its CPU rendering for ordinary local builds.
Render with `backend=cpu|cuda|rocm|vulkan` and one PEP 440 `version` (default
`0.1.0`); CPU extras pin that exact version of each backend distribution.
For example, using the canonical CPU sdist extracted into `python-dist`:

```sh
python -m pip install jinja2
backend=cuda  # or rocm, vulkan
project="build/python-$backend"
package="clap_cpp_$backend"
mkdir -p "$project/$package"
cp python-dist/{README.md,LICENSE,pyproject.toml.j2} "$project/"
cp python-dist/clap_cpp/{clap.py,clap.pyi,py.typed} "$project/$package/"
printf '"""Native %s runtime; import clap_cpp for the public API."""\n' "$backend" > "$project/$package/__init__.py"
BACKEND="$backend" PROJECT="$project" python - <<'PY'
import os
from pathlib import Path
from jinja2 import Environment, StrictUndefined
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # install tomli on Python 3.9/3.10
source = Path('python-dist')
version = tomllib.loads((source / 'pyproject.toml').read_text())['project']['version']
template = Environment(undefined=StrictUndefined, keep_trailing_newline=True).from_string(
    (source / 'pyproject.toml.j2').read_text())
(Path(os.environ['PROJECT']) / 'pyproject.toml').write_text(
    template.render(backend=os.environ['BACKEND'], version=version))
PY
cp -R "libraries/linux-x64-$backend/clap_cpp/." "$project/$package/"
cp "$project/LICENSE" "$project/$package/licenses/clap-cpp.txt"
python -m build --wheel "$project"
```

The release workflow performs these copies and Jinja renders directly; no
project-generation or staging script is needed. The selector still uses
`clap_cpp_cuda`, `clap_cpp_rocm` and `clap_cpp_vulkan` with unchanged public APIs.

## Development and source distributions

```sh
python -m build --sdist python
# Unpack the archive, then explicitly stage its external runtime:
tar -xf python/dist/clap_cpp-0.1.0.tar.gz -C build
cp -R libraries/linux-x64/clap_cpp/. build/clap_cpp-0.1.0/clap_cpp/
cp build/clap_cpp-0.1.0/LICENSE build/clap_cpp-0.1.0/clap_cpp/licenses/clap-cpp.txt
python -m build --wheel build/clap_cpp-0.1.0
python -m pip install 'build/clap_cpp-0.1.0/dist/'*.whl
python -m pip install pytest mypy typing_extensions build 'scikit-build-core>=1.1.2' jinja2
python -m pytest python/tests
```

Source distributions contain wrapper/typing files, `LICENSE` and
metadata/template files, not native C++ snapshots, staged binaries, build hooks
or native build definitions. Tests are excluded from both wheels and sdists; release
CI supplies them separately from the exact release checkout and tests the installed
wheel on each supported interpreter. Plain `python -m build python` builds from an
unstaged sdist and therefore produces an unusable runtime-free wheel; do not
publish it. Use the explicit `--sdist`, unpack, stage, `--wheel` sequence above.
Backend projects can also be rendered from the extracted CPU sdist before staging
their runtimes. No NumPy, Python development headers or pybind11 are required.

## Releases

Release packaging consumes independently built native runtime artifacts, stages
each into the extracted wrapper source project, builds `py3-none` platform wheels,
and validates/repairs them before publication. GPU driver and external runtime
requirements remain those listed above; packaging is not a GPU runtime test.

Stable releases publish the four distributions using PyPI OIDC. Configure a PyPI
trusted publisher (owner `gorse-io`, repository `clap.cpp`, workflow `release.yml`,
environment `pypi`) for **each** of `clap-cpp`, `clap-cpp-cuda`, `clap-cpp-rocm` and
`clap-cpp-vulkan`, and any required GitHub `pypi` environment approvals.
Prereleases only attach assets. Normalize the release version and the three exact
extra pins together by rendering the shared Jinja template for each backend.
