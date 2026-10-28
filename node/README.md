# clap.cpp Node.js bindings

[![npm](https://img.shields.io/npm/v/clap-cpp)](https://www.npmjs.com/package/clap-cpp)
![NPM Type Definitions](https://img.shields.io/npm/types/clap-cpp)
[![npm downloads](https://img.shields.io/npm/dm/clap-cpp)](https://www.npmjs.com/package/clap-cpp)

Requires Node.js 22+.

## Install

```sh
npm install clap-cpp
```

## Usage

```js
const { CLAP } = require('clap-cpp');

const model = new CLAP('larger_clap_general-q4_k_m.gguf');
try {
  const textEmbedding = model.encodeText('Sound of a dog');
  const audioEmbedding = model.encodeAudio('dog.wav');
} finally {
  model.close();
}
```

For EmbeddingGemma 2 audio, load the matching audio mmproj:

```js
const model = new CLAP('embeddinggemma2.gguf', { threads: 4, mmproj: 'mmproj-Q8_0.gguf' });
const embedding = model.encodeAudio('recording.wav');
model.close();
```

Numeric `threads` is still accepted. Text works without `mmproj`; missing or
incompatible projectors throw native errors. Image encoding is unsupported.
