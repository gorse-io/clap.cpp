# clap.cpp Node.js bindings

[![npm](https://img.shields.io/npm/v/clap-cpp)](https://www.npmjs.com/package/clap-cpp)
[![npm downloads](https://img.shields.io/npm/dm/clap-cpp)](https://www.npmjs.com/package/clap-cpp)

## Install

```sh
npm install clap-cpp
```

Requires Node.js 22+. The `clap-cpp` package contains JavaScript, TypeScript
declarations and tests, not native runtimes. npm selects a platform package from
its exact-version `optionalDependencies`:

| Operating system | Architecture | Optional native package |
| --- | --- | --- |
| Linux (glibc) | x64 | `@gorse/clap-cpp-linux-x64` |
| Linux (glibc) | ARM64 | `@gorse/clap-cpp-linux-arm64` |
| macOS 15+ | Apple Silicon (ARM64) | `@gorse/clap-cpp-darwin-arm64` |
| Windows | x64 | `@gorse/clap-cpp-win32-x64` |

Keep optional dependencies enabled; do not install with `--omit=optional` or
`--no-optional`. Each native package must have the same version as `clap-cpp`.
Linux native manifests declare `libc: ["glibc"]` as well as `os` and `cpu`;
other native manifests declare their matching `os` and `cpu`.

Linux release runtimes are built in a manylinux environment and require glibc
plus the system C++ runtime (`libstdc++`). musl-based distributions such as Alpine
are not supported. macOS uses the common library build's deployment target.
Cross-platform checks do not establish compatibility with older operating systems.

Consumers do not need CMake, a C++ compiler, a Git checkout, Python, or FFmpeg.
There are no clap-cpp install, build, rebuild, or prepack hooks, and the npm
package does not contain the native source tree. Unsupported platforms or
architectures are rejected. Missing optional packages report the required
package/version and suggest reinstalling with `npm install --include=optional`.
A version mismatch or missing runtime is an error, not a request to compile,
load a legacy runtime directory, or search for a system library.

The JavaScript wrapper calls the public C API through Koffi. There is no
clap.cpp-specific C++ Node addon or separate `node/CMakeLists.txt`. Koffi remains
a native FFI dependency; this is not a native-code-free package. The shared core
statically links bundled ggml and ICU. The macOS runtime includes Metal with an
embedded shader library; Linux and Windows runtimes are CPU-only. OpenMP, other
GPU backends and host-specific CPU instruction flags are disabled for
portability. The wrapper resolves the matching native package's `package.json`
with `require.resolve`, checks its version, then loads only that package's
`runtime/clap_cpp/libclap.so`, `runtime/clap_cpp/libclap.dylib`, or
`runtime/clap_cpp/clap.dll`. Native packages preserve the common runtime layout,
including companion libraries and dependency license notices. They do not include Python code or require Python at runtime.

## Building and packaging from a checkout

The native library is built **once** in the external `build-libraries` stage,
using the repository-root CMake graph. Python and npm package the same artifact;
neither language's packaging step compiles C++. Ordinary CI runs the C++, Python
and Node tests together in one platform matrix job.

For example, on a Linux x64 host, run these commands from the repository root:

```sh
git submodule update --init --recursive
cmake --preset libraries
cmake --build build/libraries --target clap --parallel 4
cmake --install build/libraries --component runtime --prefix "$PWD/libraries/linux-x64"
python3 .github/scripts/pack-node.py --runtime-root libraries --output node/dist --target linux-x64
cd node
npm ci --ignore-scripts
mkdir -p artifacts
for package in dist/*; do npm pack "./$package" --pack-destination artifacts; done
npm install --no-save --package-lock=false --ignore-scripts ./artifacts/gorse-clap-cpp-linux-x64-*.tgz
npm run test:fixtures
npm test
npm run test:types
npm run test:packaging
```

For macOS, configure with `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON` as the
shared CI action does. Windows builds use a Visual Studio developer environment.
The common install puts the C library and dependency notices under `clap_cpp/`;
that namespace does not require Python. Python wheel staging and npm native
package staging each copy these same native bytes into their own distribution.

The external `.github/scripts/pack-node.py` stages the main package in
`node/dist/clap-cpp` and each native package in
`node/dist/clap-cpp-<platform>-<arch>`. Omit `--target` to require all four runtime
directories; stage packages again after source changes. Publish all native
packages first, then publish `clap-cpp` at the same version. Maintainer library
builds need CMake 3.20+ and a C++20 compiler; npm consumers do not.
Keep Koffi's optional prebuilt dependencies enabled, even in maintainer installs;
`--omit=optional` would also remove its platform binaries.

Packaging tests inspect real main and native tarballs, install both explicitly
outside the checkout, generate fixtures using the installed
`test/tiny_models.py`, and run installed runtime tests. They require Python 3
(override the interpreter with `PYTHON`). Stage packages again after source
changes. `CLAP_CPP_PACKAGE_DIR` and `CLAP_CPP_NATIVE_PACKAGE_DIR` can override the
staged main and host-native directories. In an installed package, the defaults
are the installed main package and its resolved native dependency, so repacking
does not require the checkout's staging script.

By default packaging tests require the host native package. Set
`CLAP_CPP_REQUIRE_ALL_RUNTIMES=1` for release validation to inspect all four
native packages and their licenses; each runtime still needs matching-host
inference testing. Local explicit-tarball installation does **not** prove registry
selection. `.github/scripts/test-node-install.py` instead serves the actual
platform tarballs through a temporary local scoped registry and asserts that npm
downloads only the host runtime; release CI runs it against all four real native
packages. This is not a publication to the public npm registry. Tests and fixture
generation are maintainer operations only; npm consumers do not need Python.

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
