# clap.cpp Node.js bindings

[![npm](https://img.shields.io/npm/v/clap-cpp)](https://www.npmjs.com/package/clap-cpp)
[![npm downloads](https://img.shields.io/npm/dm/clap-cpp)](https://www.npmjs.com/package/clap-cpp)

## Install

```sh
npm install clap-cpp
```

Requires Node.js 22+. The universal npm tarball includes prebuilt shared
libraries for these platform/architecture pairs:

| Operating system | Architecture | Runtime directory |
| --- | --- | --- |
| Linux (glibc) | x64 | `runtime/linux-x64` |
| Linux (glibc) | ARM64 | `runtime/linux-arm64` |
| macOS 15+ | Apple Silicon (ARM64) | `runtime/darwin-arm64` |
| Windows | x64 | `runtime/win32-x64` |

Linux release builds use Ubuntu 22.04 and require glibc and the system C++
runtime (`libstdc++`); musl-based distributions such as Alpine are not supported.
There is no musl binary or source-build fallback. The macOS deployment target is
explicitly set to 15.0, rather than inheriting the build runner's default.
Cross-platform release tests run on the declared build platforms; they do not
establish compatibility with older Linux distributions.

Consumers do not need CMake, a C++ compiler, a Git checkout, Python, or FFmpeg.
There are no clap-cpp install, build, rebuild, or prepack hooks, and the npm
package does not contain the native source tree. Unsupported platforms or
architectures are rejected; a missing runtime is an incomplete package, not a
request to compile or search for a system library.

The JavaScript wrapper calls the public C API through Koffi. There is no
clap.cpp-specific C++ Node addon or separate `node/CMakeLists.txt`. Koffi remains
a native FFI dependency; this is not a native-code-free package. The shared core
statically links bundled ggml and ICU. The macOS runtime includes Metal with an
embedded shader library; Linux and Windows runtimes are CPU-only. OpenMP, other
GPU backends and host-specific CPU instruction flags are disabled for
portability. The core is loaded only from
`runtime/<process.platform>-<process.arch>/` in this installation. Dependency
license notices are installed in each runtime directory's `licenses/` subdirectory.

## Building and packaging from a checkout

Native builds are a separate maintainer step, using the repository-root CMake
graph, CMake 3.20+ and a C++20 compiler. Initialize the pinned submodules first.
For example, on a Linux x64 host, run these commands from the repository root:

```sh
git submodule update --init --recursive
cmake --preset node-linux-x64 -B build/node-linux-x64
cmake --build build/node-linux-x64 --config Release --target clap --parallel 4
cmake --install build/node-linux-x64 --config Release --component runtime \
  --prefix "$PWD/node/runtime/linux-x64"
cp LICENSE node/LICENSE
cp python/tests/tiny_models.py node/test/tiny_models.py
cd node
npm ci
npm run test:fixtures
npm test
npm run test:types
npm run test:packaging
npm pack
```

The root `node-linux-x64`, `node-linux-arm64`, `node-darwin-arm64`, and
`node-win32-x64` presets carry the complete shared-library, static-dependency and
portability settings, including a flat CMake install layout. Build each runtime
on its matching host/toolchain, then collect the four installed directories into
`node/runtime/` before publishing one universal tarball. Linux presets select
native ARM64/x64 hosts; they do not configure cross-compilers. On Windows, use
PowerShell `Copy-Item` instead of `cp` and an absolute install prefix. No npm step
builds or vendors the core library.

Packaging tests create and inspect a real tarball, install it outside the
checkout, generate fixtures using its bundled `test/tiny_models.py`, and run its
runtime tests. They require Python 3 (override the interpreter with `PYTHON`).
By default they require the host runtime and validate every included runtime's
licenses, so a single-platform checkout can be tested. Release validation must
set `CLAP_CPP_REQUIRE_ALL_RUNTIMES=1` when running `npm run test:packaging` to
require all four runtime directories. Tests and fixture generation are maintainer
operations only; npm consumers do not need Python.

## Usage

```js
const { CLAP } = require('clap-cpp');

const model = new CLAP('/path/to/model.gguf', 4);
try {
  const text = model.encodeText('a piano playing'); // owned Float32Array
  const audio = model.encodeAudio('/path/to/audio.wav');
  // CLAP projections are unit-normalized, so their dot product is cosine.
  const similarity = text.reduce((sum, x, i) => sum + x * audio[i], 0);
  console.log(similarity);
} finally {
  model.close(); // optional early release; garbage collection also frees weights
}
```

The package exports only `CLAP`. Its inference API consists of:

- `encodeText(textOrTokens)`: accepts a string, `Int32Array` or `number[]` token IDs.
- `encodeAudio(path, cropOffset = 0)`: decodes WAV/MP3/FLAC by contents, downmixes to mono and automatically resamples for the model.
- `encodeAudio(pcm, sampleRate = 48000, cropOffset = 0)`: accepts mono `Float32Array` samples, with automatic resampling.

Both methods return owned `Float32Array` embeddings. Crop offsets are measured in samples; support depends on the model architecture. `close()` is idempotent; encoding after closing throws.

Inference is synchronous and blocks the calling JavaScript thread. Use a separate model per worker for parallel execution. Tokenization, audio decoding, resampling, frame features and model metadata are not separate public APIs.
