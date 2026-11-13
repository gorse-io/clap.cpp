'use strict';
const path = require('node:path');
const { types } = require('node:util');
const { Worker } = require('node:worker_threads');
const { defaults, create, destroy, text, tokens, audio, audioFile, maxSize, check, embedding } = require('./_native');
const workerPath = path.join(__dirname, '_encode-worker.js');
const maxInt = 2147483647;
const typedArrayPrototype = Object.getPrototypeOf(Float32Array.prototype);
const lengthOf = Function.call.bind(Object.getOwnPropertyDescriptor(typedArrayPrototype, 'length').get);
const copyInto = Function.call.bind(typedArrayPrototype.set);
const isFloat32Array = types.isFloat32Array;
const isInt32Array = types.isInt32Array;
const FloatArray = Float32Array;
const IntArray = Int32Array;

function integer(value, name, min, max) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
    throw new TypeError(`${name} is outside the supported integer range`);
  }
  return value;
}
function string(value, name, nonempty = false) {
  if (typeof value !== 'string' || (nonempty && !value.length) || value.includes('\0')) {
    throw new TypeError(`${name} must be ${nonempty ? 'nonempty and ' : ''}a string containing no NUL bytes`);
  }
  return value;
}
function tokenCopy(value) {
  let copy;
  if (isInt32Array(value)) {
    const length = lengthOf(value);
    if (!length) throw new TypeError('tokens must be nonempty');
    copy = new IntArray(length);
    copyInto(copy, value);
    for (let i = 0; i < length; ++i) integer(copy[i], 'token ID', 0, maxInt);
  } else if (Array.isArray(value)) {
    // Finish all Proxy/getter reads before taking a handle lease: they can close.
    const length = integer(value.length, 'token count', 1, Math.min(maxSize, maxInt));
    copy = new IntArray(length);
    for (let i = 0; i < length; ++i) copy[i] = integer(value[i], 'token ID', 0, maxInt);
  } else {
    throw new TypeError('text must be a string, Int32Array or number[]');
  }
  return copy;
}
function pcmCopy(value) {
  if (!isFloat32Array(value)) throw new TypeError('PCM must be a nonempty Float32Array of mono samples');
  const length = lengthOf(value);
  if (!length || length > 268435456) throw new TypeError('PCM must be nonempty and at most 1 GiB');
  const copy = new FloatArray(length);
  copyInto(copy, value);
  for (let i = 0; i < length; ++i) {
    if (!Number.isFinite(copy[i])) throw new TypeError('PCM samples must be finite');
  }
  return copy;
}
function audioInput(value, sampleRateOrOffset, cropOffset) {
  if (typeof value === 'string') {
    const filename = string(value, 'path', true);
    const offset = integer(sampleRateOrOffset === undefined ? 0 : sampleRateOrOffset, 'cropOffset', 0, maxSize);
    return ['audioFile', [filename, offset]];
  }
  const pcm = pcmCopy(value);
  const rate = integer(sampleRateOrOffset === undefined ? 48000 : sampleRateOrOffset, 'sampleRate', 1, maxInt);
  const offset = integer(cropOffset, 'cropOffset', 0, maxSize);
  return ['audio', [pcm, lengthOf(pcm), rate, offset]];
}
function textInput(value) {
  const input = typeof value === 'string' ? string(value, 'text') : tokenCopy(value);
  return typeof input === 'string' ? ['text', [input]] : ['tokens', [input, lengthOf(input)]];
}
const encoders = { text, tokens, audio, audioFile };
function destroyIfIdle(state) {
  if (!state.closed || state.active !== 0) return;
  const pointer = state.pointer;
  state.pointer = null;
  if (pointer !== null) destroy(pointer);
}
function release(state) {
  state.closed = true;
  destroyIfIdle(state);
}
function borrow(state) {
  if (state.closed) throw new Error('CLAP is closed');
  ++state.active;
  return state.pointer;
}
function giveBack(state) {
  --state.active;
  destroyIfIdle(state);
}
const finalizer = new FinalizationRegistry(release);
function encode(state, operation, args) {
  const pointer = borrow(state);
  try { return embedding(encoders[operation], [pointer, ...args]); }
  finally { giveBack(state); }
}
function encodeAsync(state, operation, args) {
  const pointer = borrow(state);
  return new Promise((resolve, reject) => {
    let worker;
    try {
      // Koffi's official pointer representation is BigInt, not an isolate-local
      // external object: https://koffi.dev/pointers . No model copy or new handle.
      worker = new Worker(workerPath, { workerData: { pointer, operation, args } });
    } catch (error) {
      giveBack(state);
      reject(error);
      return;
    }
    let result;
    let failure;
    worker.once('message', message => { result = message; });
    worker.once('error', error => { failure = error; });
    // An error/message alone does NOT prove native execution has stopped.
    // Retain the lease until exit, including worker bootstrap/error paths.
    worker.once('exit', code => {
      giveBack(state);
      if (failure) return reject(failure);
      if (code !== 0) return reject(new Error(`CLAP worker exited with code ${code}`));
      if (!result) return reject(new Error('CLAP worker exited without a result'));
      if (result.error) {
        if ((result.error.name !== 'TypeError' && result.error.name !== 'Error') || typeof result.error.message !== 'string') {
          return reject(new Error('Malformed CLAP worker error'));
        }
        const ErrorType = result.error.name === 'TypeError' ? TypeError : Error;
        return reject(new ErrorType(result.error.message));
      }
      if (!isFloat32Array(result.value)) return reject(new Error('Malformed CLAP worker result'));
      resolve(result.value);
    });
  });
}

class CLAP {
  #state;
  #token;

  constructor(modelPath, threads = 0) {
    const filename = string(modelPath, 'path', true);
    const threadCount = integer(threads, 'threads', 0, maxInt);
    const options = defaults();
    options.threads = threadCount;
    const output = [null];
    const state = { pointer: null, active: 0, closed: false };
    const token = Object.create(null);
    try {
      check(create(filename, options, output));
      state.pointer = output[0];
      this.#state = state;
      this.#token = token;
      // Neither finalizer nor lease state refers back to its JS owner.
      finalizer.register(this, state, token);
    } catch (error) {
      if (output[0] !== null) destroy(output[0]);
      state.pointer = null;
      state.closed = true;
      throw error;
    }
  }

  encodeText(value) {
    // Brand-check before user-controlled input reads, then lease AFTER them.
    const state = this.#state;
    return encode(state, ...textInput(value));
  }

  encodeAudio(value, sampleRateOrOffset, cropOffset = 0) {
    const state = this.#state;
    return encode(state, ...audioInput(value, sampleRateOrOffset, cropOffset));
  }

  async encodeTextAsync(value) {
    const state = this.#state;
    return encodeAsync(state, ...textInput(value));
  }

  async encodeAudioAsync(value, sampleRateOrOffset, cropOffset = 0) {
    const state = this.#state;
    return encodeAsync(state, ...audioInput(value, sampleRateOrOffset, cropOffset));
  }

  close() {
    const state = this.#state;
    finalizer.unregister(this.#token);
    release(state);
  }
}
module.exports = { CLAP };
