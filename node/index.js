'use strict';
const path = require('node:path');
const { types } = require('node:util');

// Resolve only the matching optional npm package. Never search PATH, the
// system linker path, or a legacy package-local runtime.
const target = `${process.platform}-${process.arch}`;
const libraryName = {
  'linux-x64': 'libclap.so',
  'linux-arm64': 'libclap.so',
  'darwin-arm64': 'libclap.dylib',
  'win32-x64': 'clap.dll',
}[target];
if (!libraryName) throw new Error(`Unsupported platform or architecture: ${target}`);
const packageName = `@gorse/clap-cpp-${target}`;
const { version } = require('./package.json');
let nativeManifestPath;
try {
  nativeManifestPath = require.resolve(`${packageName}/package.json`);
} catch (error) {
  throw new Error(`Missing optional native package ${packageName}@${version} for ${target}. Reinstall clap-cpp with optional dependencies enabled (npm install --include=optional); do not use --omit=optional or --no-optional.`, { cause: error });
}
const nativeManifest = require(nativeManifestPath);
if (nativeManifest.version !== version) {
  throw new Error(`Native package version mismatch: clap-cpp@${version} requires ${packageName}@${version}, but found ${nativeManifest.version}. Reinstall matching package versions.`);
}
const runtime = path.join(path.dirname(nativeManifestPath), 'runtime', libraryName);
if (!require('node:fs').existsSync(runtime)) {
  throw new Error(`Missing prebuilt clap runtime in ${packageName}@${version}: ${runtime}. Reinstall the optional native package; no source-build or system-library fallback is available.`);
}
const koffi = require('koffi');
const lib = koffi.load(runtime);

// Anonymous types keep independently installed copies from colliding in Koffi's
// named type registry. Enums are C int; struct layout uses the platform C ABI.
const handle = koffi.pointer(koffi.opaque());
const optionsType = koffi.struct({ threads: 'int', device: 'str', activation_precision: 'int' });
const defaults = lib.func('clap_default_options', optionsType, []);
const lastError = lib.func('clap_last_error', 'str', []);
const create = lib.func('clap_create', 'int', ['str', koffi.pointer(optionsType), koffi.out(koffi.pointer(handle))]);
const destroy = lib.func('clap_destroy', 'void', [handle]);
const free = lib.func('clap_free', 'void', ['void *']);
const outputs = [koffi.out(koffi.pointer('float', 2)), koffi.out(koffi.pointer('size_t'))];
const text = lib.func('clap_encode_text', 'int', [handle, 'str', ...outputs]);
const tokens = lib.func('clap_encode_tokens', 'int', [handle, 'const int *', 'size_t', ...outputs]);
const audio = lib.func('clap_encode_audio', 'int', [handle, 'const float *', 'size_t', 'int', 'size_t', ...outputs]);
const audioFile = lib.func('clap_encode_audio_file', 'int', [handle, 'str', 'size_t', ...outputs]);
const maxInt = 2147483647;
const maxSize = koffi.sizeof('size_t') === 4 ? 4294967295 : Number.MAX_SAFE_INTEGER;
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
    // Read length once: array Proxy traps and index getters can close a model or
    // mutate the array. Finish every such read before borrowing its handle.
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
function check(status) {
  if (status === 0) return;
  // Capture TLS immediately, before any other fallible call can clear it.
  const message = lastError() || `CLAP failed (status ${status})`;
  throw status === 1 ? new TypeError(message) : new Error(message);
}
function release(state) {
  const pointer = state.pointer;
  state.pointer = null;
  if (pointer !== null) destroy(pointer);
}
const finalizer = new FinalizationRegistry(release);

function embedding(fn, args) {
  const values = [null];
  const count = [0];
  try {
    check(fn(...args, values, count));
    const length = integer(count[0], 'native embedding length', 0, maxSize);
    if (!length) return new FloatArray(0);
    // decode returns an owning typed array, not a view of the C allocation.
    // The finally also runs if decoding or JS allocation throws.
    return koffi.decode(values[0], 'float', length);
  } finally {
    if (values[0] !== null) free(values[0]);
  }
}

class CLAP {
  #state;
  #token;

  constructor(modelPath, threads = 0) {
    const filename = string(modelPath, 'path', true);
    const threadCount = integer(threads, 'threads', 0, maxInt);
    const options = defaults(); // Returned by value, not a pointer.
    options.threads = threadCount;
    const output = [null];
    const state = { pointer: null };
    const token = Object.create(null);
    try {
      check(create(filename, options, output));
      state.pointer = output[0];
      this.#state = state;
      this.#token = token;
      // Held state never refers to this. Each owner gets a unique unregister
      // token, and failed registration cannot leak a successful native create.
      finalizer.register(this, state, token);
    } catch (error) {
      if (output[0] !== null) destroy(output[0]);
      state.pointer = null;
      throw error;
    }
  }

  encodeText(value) {
    // Brand-check before running user code; Object.create/copies/Proxy receivers
    // must never gain ownership or borrow another object's raw pointer.
    const state = this.#state;
    const input = typeof value === 'string' ? string(value, 'text') : tokenCopy(value);
    if (state.pointer === null) throw new Error('CLAP is closed');
    return typeof input === 'string'
      ? embedding(text, [state.pointer, input])
      : embedding(tokens, [state.pointer, input, lengthOf(input)]);
  }

  encodeAudio(value, sampleRateOrOffset, cropOffset = 0) {
    const state = this.#state;
    if (typeof value === 'string') {
      const filename = string(value, 'path', true);
      const offset = integer(sampleRateOrOffset === undefined ? 0 : sampleRateOrOffset, 'cropOffset', 0, maxSize);
      if (state.pointer === null) throw new Error('CLAP is closed');
      return embedding(audioFile, [state.pointer, filename, offset]);
    }
    const pcm = pcmCopy(value);
    const rate = integer(sampleRateOrOffset === undefined ? 48000 : sampleRateOrOffset, 'sampleRate', 1, maxInt);
    const offset = integer(cropOffset, 'cropOffset', 0, maxSize);
    if (state.pointer === null) throw new Error('CLAP is closed');
    return embedding(audio, [state.pointer, pcm, lengthOf(pcm), rate, offset]);
  }

  close() {
    const state = this.#state;
    finalizer.unregister(this.#token);
    release(state);
  }
}
module.exports = { CLAP };
