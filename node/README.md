# clap.cpp Node.js bindings

[![npm](https://img.shields.io/npm/v/clap-cpp)](https://www.npmjs.com/package/clap-cpp)
[![npm downloads](https://img.shields.io/npm/dm/clap-cpp)](https://www.npmjs.com/package/clap-cpp)

## Install

```sh
npm install clap-cpp
```

Requires Node.js 22+. The `clap-cpp` package contains JavaScript, TypeScript
declarations, README and license, not native runtimes or tests. npm selects a platform package from
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

The four platform packages are defined in `node/packages/<target>/package.json`
and registered by the root `workspaces` field. Their manifests are tracked in Git;
compiled runtimes and copied licenses are staged before packing.

For example, on a Linux x64 host, run these commands from the repository root:

```sh
git submodule update --init --recursive
cmake --preset libraries
cmake --build build/libraries --target clap --parallel 4
cmake --install build/libraries --component runtime --prefix "$PWD/node/packages/linux-x64/runtime"
cd node
cp ../LICENSE LICENSE
cp ../LICENSE packages/linux-x64/LICENSE
cp ../LICENSE packages/linux-x64/runtime/clap_cpp/licenses/clap.txt
npm ci --force --ignore-scripts
npm test
mkdir -p artifacts
npm pack --workspace packages/linux-x64 --include-workspace-root --pack-destination artifacts
python3 ../.github/scripts/validate-node-packages.py --artifacts artifacts
```

For macOS, use `darwin-arm64` and configure with
`-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`. Windows builds use a Visual Studio
developer environment and the `win32-x64` workspace. The common install puts the
C library and dependency notices under `runtime/clap_cpp/` in each workspace.
An existing `libraries/<target>/` artifact can also be copied into that directory.

Maintainer installs use `--force` because npm otherwise rejects workspaces for
other operating systems or architectures. This links the local packages without
requiring them to exist on the npm registry. Consumers use ordinary
`npm install clap-cpp`; the published optional dependencies retain their platform
constraints. Keep Koffi's optional prebuilt dependencies enabled;
`--omit=optional` would also remove its platform binaries.

For a release, stage all four runtimes and licenses, then run
`npm pack --workspaces --include-workspace-root --pack-destination artifacts`.
The release version script updates the root, workspace manifests and lockfile
together. Publish all native packages before publishing `clap-cpp` at the same
version. Maintainer library builds need CMake 3.20+ and a C++20 compiler;
npm consumers do not.

`.github/scripts/validate-node-packages.py` checks runtime payloads, dependency
licenses, platform constraints and matching versions without installing the
archives. It also verifies that the main package contains only the binding,
type declarations, README, license and package manifest. Pass `--require-all`
to require all four platform archives.

CI runs the CLAP inference test directly against the checkout and its platform
workspace. Tests and audio fixtures are not included in published packages.
The CLAP test downloads
`larger_clap_general-q4_k_m.gguf` from Hugging Face into the repository root.
It compares `dog.wav` with dog and vacuum-cleaner descriptions, using the same
softmax scores and thresholds as `rust/tests/clap.rs`.
No test publishes to the public npm registry.

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
