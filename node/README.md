# clap.cpp Node.js bindings

[![npm](https://img.shields.io/npm/v/clap-cpp)](https://www.npmjs.com/package/clap-cpp)
[![npm downloads](https://img.shields.io/npm/dm/clap-cpp)](https://www.npmjs.com/package/clap-cpp)

## Install

```sh
npm install clap-cpp
```

Requires Node.js 22+, CMake 3.20+ and a C++20 compiler. Installation builds the
package-local shared `clap` library using the repository-root CMake graph in
core-only mode. Source packages include pinned native sources and licenses;
no Git checkout, Node.js development headers, Python, or FFmpeg are needed to
compile the library.

The JavaScript wrapper calls the public C API through Koffi. There is no
clap.cpp-specific C++ Node addon or separate `node/CMakeLists.txt`. Koffi remains
a native FFI dependency; this is not a native-code-free package. The shared core
statically links bundled ggml and ICU. On macOS, Metal is enabled by default and
its shader library is embedded in the shared library. Other platforms default to
CPU-only builds. OpenMP, other GPU backends and host-specific CPU instruction
flags remain disabled for portability. The core is loaded only from this
package's own runtime directory, without searching for a system `libclap`.

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

For EmbeddingGemma 2 audio, load the matching audio mmproj:

```js
const model = new CLAP('embeddinggemma2.gguf', { threads: 4, mmproj: 'mmproj-Q8_0.gguf' });
const embedding = model.encodeAudio('recording.wav');
model.close();
```

Numeric `threads` is still accepted. Text works without `mmproj`; missing or
incompatible projectors throw native errors. Image encoding is unsupported.
