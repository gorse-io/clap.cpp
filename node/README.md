# clap.cpp for Node.js

Synchronous CPU bindings to the existing C++ `clap::Model` API, implemented with
node-addon-api and Node-API 8. No C API branch, Python runtime, GPU, OpenMP runtime,
or separately installed ICU is required for inference. TypeScript declarations
are included. The package is CommonJS; ESM can use its default export.

## Requirements and installation

- Node.js **22 or newer**, npm, CMake **3.20+**, and a C/C++17 compiler.
- Linux: Make or Ninja; macOS: Xcode command-line tools; Windows: Visual Studio
  C++ build tools. Only Linux x64 has been exercised locally; the other platforms
  use portable CMake/Node-API integration but are not locally verified.
- Python 3.9+ is needed **only to generate test fixtures**. NumPy is optional and
  enables an independent numerical text oracle.

From a checkout with its pinned submodules initialized:

```sh
git submodule update --init --recursive
cd node
npm ci                     # builds the addon via cmake-js
npm run test:fixtures      # deterministic tiny GGUF models + WAV, no downloads
npm test
npm run test:types
```

`npm install`/`npm ci` compiles from source; no prebuilt binaries are shipped.
cmake-js may download the matching Node headers on first use. CMake itself needs
no network access once the package and Node headers are available. The build
bundles static PIC CPU ggml, JSON, ICU Unicode/regex and miniaudio sources. Host-
specific x86 instruction extensions and GPU backends are disabled by default.

To override generator or build parallelism explicitly:

```sh
npm run build -- --generator 'Unix Makefiles' --parallel 2
```

## Usage

```js
const { Model, decodeAudio, resampleAudio } = require('clap-cpp');

const model = new Model('/path/to/model.gguf', 4);
try {
  console.log(model.projectionDimension, model.audioSampleRate);
  const text = model.text('a piano playing'); // owned Float32Array
  const { samples, sampleRate } = decodeAudio('/path/to/audio.wav');
  const audio = model.audio(samples, sampleRate); // automatically resampled
  // CLAP projections are unit-normalized, so their dot product is cosine.
  const similarity = text.reduce((sum, x, i) => sum + x * audio[i], 0);
  console.log(similarity);
} finally {
  model.close(); // optional early release; garbage collection also frees weights
}
```

Every inference, decode and resample call **blocks the JavaScript thread**. Use
`worker_threads` for servers/UI workloads, creating a separate model inside each
worker. Models are not transferable or shared across workers. There is no async
API, batching, similarity helper or network/model downloader in this binding.

## API

| API | Result / defaults |
| --- | --- |
| `new Model(path, threads = 0)` | Load a GGUF; zero threads selects the native default. |
| `model.projectionDimension` | Read-only embedding width. |
| `model.audioSampleRate` | Read-only model-native sample rate. |
| `model.tokenize(text, maxLength = 512)` | Owned `Int32Array`. Native tokenizer/truncation behavior applies. |
| `model.text(textOrTokens)` | Owned `Float32Array`; accepts a string, `Int32Array` or `number[]` token IDs. |
| `model.audio(pcm, sampleRate = 48000, cropOffset = 0)` | Owned embedding `Float32Array`; mono PCM is automatically resampled. Crop offset is in samples, per the native model contract. |
| `model.audioFrames(pcm, sampleRate = 48000)` | `{ values: Float32Array, frames: number, dimension: number }`. |
| `model.clamp3Features(pcm, sampleRate = 24000)` | Same frame object, with MERT-v1 all-layer means for CLaMP3. |
| `model.close()` | Idempotent early release. Other methods and property reads throw after close. |
| `decodeAudio(path)` | `{ samples: Float32Array, sampleRate: number }`; WAV/MP3/FLAC detected by contents, channels averaged to mono, source rate preserved. |
| `resampleAudio(pcm, sourceRate, targetRate = 48000)` | Owned mono `Float32Array`; same-rate input is copied unchanged. |

Frame values are flat **time-major**: `values[t * dimension + d]`, with exactly
`frames * dimension` elements. All returned typed arrays own their memory and
remain valid after subsequent inference and after `close()`.

PCM must be a nonempty **Float32Array** with finite values and at most 1 GiB of
samples. No coercion from ordinary arrays or Float64Array is performed. Sample
rates must be integers in `[1, 2147483647]`; threads in `[0, 2147483647]`;
`maxLength` in `[1, 2147483647]`. Token IDs must be nonnegative signed 32-bit
integers. Crop offsets must be nonnegative safe integers representable by native
`size_t`. Paths must be nonempty strings without embedded NUL bytes. Invalid
arguments and native failures throw JavaScript exceptions; invalid-argument
errors generally use `TypeError`. `undefined` for an optional argument selects
its default; `null` does not. Resource limits/context lengths remain model-
dependent; an integer within the API range is not a promise that arbitrary huge
requests can be allocated or inferred.

### Architecture behavior

Architecture selection, preprocessing, pooling, context lengths and normalization
are the native C++ implementation's behavior, not reimplemented in JavaScript:

- CLAP text/audio outputs are unit-normalized projections.
- CLaMP3 outputs are native **unnormalized** projections and require a bundled
  MERT-v1 frontend. Do not treat their raw dot product as cosine similarity.
- MERT-v1, MERT-v2 and MuQ expose final-layer `audioFrames`; `audio` mean-pools
  frames. `clamp3Features` is specifically MERT-v1's all-layer, five-second-window
  representation, not its final-layer frames; it requires at least one second.
- MuQ-MuLan supports joint embeddings, not frame/feature APIs or crop offsets;
  audio follows native per-clip normalization/averaging without final normalization.
- Unsupported modalities/methods throw. The tokenizer's default maximum is not a
  universal model context size; use the model-specific supported length when
  requesting tokens for inference.

Resampling uses the native causal linear + anti-alias low-pass implementation,
not zero-phase sinc interpolation. Startup latency is expected. Output duration
is `ceil(input.length * targetRate / sourceRate)`.

## Standalone npm tarballs

```sh
mkdir -p artifacts
npm run test:packaging
npm pack --pack-destination artifacts
npm install --prefix installed ./artifacts/clap-cpp-0.1.0.tgz
CLAP_CPP_MODULE="$PWD/installed/node_modules/clap-cpp" npm test
```

The `prepack` hook copies the **pinned** native subset and third-party licenses
into ignored `native/`/`THIRD_PARTY_LICENSES/` directories. It refuses mismatched
submodule revisions. Checkout builds use the parent repository; tarball installs
build entirely from `native/`, without Git or a surrounding clap.cpp checkout.
The tarball carries a copy of the shared stdlib tiny-model generator, so even
its tests can run standalone:

```sh
cd installed/node_modules/clap-cpp
npm run test:fixtures
npm test
```

Generated native copies, fixtures, binaries, dependencies and tarballs must not
be committed. `npm run test:types` needs the development dependencies and is
normally run from the source package. The runtime regression suite covers real
CLAP text/waveform inference, MERT frame/pooling/features, lifetime/ownership,
argument validation, WAV decoding and resampling. It uses tiny deterministic
models, not published-weight accuracy or performance benchmarks.
