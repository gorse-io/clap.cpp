# clap.cpp Node.js bindings

## Install

```sh
npm install clap-cpp
```

## Usage

```js
const { Model, decodeAudio } = require('clap-cpp');

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

For EmbeddingGemma 2 audio, pass the matching mmproj in constructor options:

```js
const model = new Model('embeddinggemma2.gguf', { threads: 4, mmproj: 'mmproj-Q8_0.gguf' });
const { samples, sampleRate } = decodeAudio('recording.wav');
const embedding = model.audio(samples, sampleRate);
model.close();
```

Omitting `mmproj` preserves text/single-model loading; numeric `threads` is
still accepted. Missing/incompatible projectors throw native errors. Image
encoding is not supported.
