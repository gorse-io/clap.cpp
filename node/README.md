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
