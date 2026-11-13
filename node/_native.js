'use strict';
// Private ABI shared by the main isolate and per-operation workers.
const path = require('node:path');
const target = `${process.platform}-${process.arch}`;
const libraryName = {
  'linux-x64': 'libclap.so', 'linux-arm64': 'libclap.so',
  'darwin-arm64': 'libclap.dylib', 'win32-x64': 'clap.dll',
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
// Anonymous types avoid process-global named type registry collisions.
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
const maxSize = koffi.sizeof('size_t') === 4 ? 4294967295 : Number.MAX_SAFE_INTEGER;
const FloatArray = Float32Array;
function check(status) {
  if (status === 0) return;
  // Always read TLS synchronously in the isolate/thread that made the C call.
  const message = lastError() || `CLAP failed (status ${status})`;
  throw status === 1 ? new TypeError(message) : new Error(message);
}
function embedding(fn, args) {
  const values = [null];
  const count = [0];
  try {
    check(fn(...args, values, count));
    const length = count[0];
    if (typeof length !== 'number' || !Number.isInteger(length) || length < 0 || length > maxSize) {
      throw new TypeError('native embedding length is outside the supported integer range');
    }
    if (!length) return new FloatArray(0);
    // Owning copy; release C output even if decoding/allocation fails.
    return koffi.decode(values[0], 'float', length);
  } finally {
    if (values[0] !== null) free(values[0]);
  }
}
module.exports = { defaults, create, destroy, text, tokens, audio, audioFile, maxSize, check, embedding };
