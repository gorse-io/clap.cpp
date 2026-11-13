/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
const runtime = path.join(path.dirname(nativeManifestPath), 'runtime', 'clap_cpp', libraryName);
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
const projectionDimension = lib.func('clap_projection_dimension', 'int', [handle]);
const outputs = [koffi.out(koffi.pointer('float')), 'size_t'];
const text = lib.func('clap_encode_text', 'int', [handle, 'str', ...outputs]);
const audio = lib.func('clap_encode_audio', 'int', [handle, 'const float *', 'size_t', 'int', 'size_t', ...outputs]);
const audioFile = lib.func('clap_encode_audio_file', 'int', [handle, 'str', 'size_t', ...outputs]);
const maxInt = 2147483647;
const maxSize = koffi.sizeof('size_t') === 4 ? 4294967295 : Number.MAX_SAFE_INTEGER;
const typedArrayPrototype = Object.getPrototypeOf(Float32Array.prototype);
const lengthOf = Function.call.bind(Object.getOwnPropertyDescriptor(typedArrayPrototype, 'length').get);
const copyInto = Function.call.bind(typedArrayPrototype.set);
const isFloat32Array = types.isFloat32Array;
const FloatArray = Float32Array;

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
  const length = integer(projectionDimension(args[0]), 'native embedding length', 1, maxInt);
  const values = new FloatArray(length);
  check(fn(...args, values, length));
  return values;
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
    const input = string(value, 'text');
    if (state.pointer === null) throw new Error('CLAP is closed');
    return embedding(text, [state.pointer, input]);
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
