'use strict';
const { test, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { createRequire } = require('node:module');
const threads = require('node:worker_threads');

// Observe actual workers/native destroys, not mock inference. Fault injection
// covers startup paths without introducing a public pointer or test API.
const modulePath = process.env.CLAP_CPP_MODULE || path.join(__dirname, '..');
const localRequire = createRequire(path.join(path.resolve(modulePath), 'index.js'));
const koffi = localRequire('koffi');
const load = koffi.load;
let destroys = 0;
koffi.load = function (...args) {
  const lib = load.apply(this, args);
  const func = lib.func.bind(lib);
  return { func(...signature) {
    const fn = func(...signature);
    if (signature[0] !== 'clap_destroy') return fn;
    return (...values) => { destroys++; return fn(...values); };
  } };
};
const RealWorker = threads.Worker;
let fault;
let nextMessage;
let activeWorkers = 0;
let maxWorkers = 0;
const pointers = [];
threads.Worker = class extends RealWorker {
  constructor(filename, options) {
    const mode = fault;
    fault = undefined;
    if (mode === 'constructor') throw new Error('injected worker constructor failure');
    if (mode === 'startup') {
      filename = 'throw new Error("injected worker startup failure")';
      options = { eval: true };
    } else if (mode === 'empty') {
      filename = '';
      options = { eval: true };
    } else if (mode === 'malformed') {
      filename = 'require("node:worker_threads").parentPort.postMessage({value: [1, 2]})';
      options = { eval: true };
    } else if (mode === 'linger') {
      filename = `require(${JSON.stringify(filename)}); setTimeout(() => {}, 100);`;
      options = { ...options, eval: true };
    }
    super(filename, options);
    if (nextMessage) { this.once('message', nextMessage); nextMessage = undefined; }
    if (options.workerData) pointers.push(options.workerData.pointer);
    maxWorkers = Math.max(maxWorkers, ++activeWorkers);
    this.once('exit', () => { activeWorkers--; });
  }
};
const { CLAP } = require(modulePath);
after(() => { threads.Worker = RealWorker; koffi.load = load; });

const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'clap-node-concurrency-'));
after(() => fs.rmSync(directory, { recursive: true, force: true }));
const generator = path.resolve(__dirname, '../../tests/tiny_model.py');
const generated = spawnSync(process.env.PYTHON || 'python3', ['-B', '-c', `
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location('tiny', sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p = pathlib.Path(sys.argv[2])
f = m.write_model(p / 'model.gguf', waveform=True)
(p / 'fixture.json').write_text(json.dumps(f))
`, generator, directory], { encoding: 'utf8' });
assert.equal(generated.status, 0, generated.stderr);
const fixture = JSON.parse(fs.readFileSync(path.join(directory, 'fixture.json'), 'utf8'));
const modelPath = path.join(directory, 'model.gguf');
function near(actual, expected) {
  assert.ok(actual instanceof Float32Array);
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) < 2e-5, `${i}: ${actual[i]} != ${expected[i]}`);
  }
}
function tokenIds(entry) { return Int32Array.from([0, ...Array.from(Buffer.from(entry.input, 'utf8'), byte => byte + 5), 2]); }
function pcm() {
  return Float32Array.from({ length: 4800 }, (_, i) => 0.2 * Math.sin(i * 0.07));
}

test('async encode methods exist (baseline RED)', () => {
  assert.equal(typeof CLAP.prototype.encodeTextAsync, 'function');
  assert.equal(typeof CLAP.prototype.encodeAudioAsync, 'function');
});

test('Promise.all uses overlapping workers and one native handle; scalar text oracle', async () => {
  const model = new CLAP(modelPath);
  try {
    pointers.length = 0;
    maxWorkers = 0;
    const pending = fixture.cases.map((entry, i) => model.encodeTextAsync(i % 2 ? tokenIds(entry) : entry.input));
    assert.ok(pending.every(value => value instanceof Promise));
    assert.ok(maxWorkers > 1, 'wrapper must not queue calls serially');
    assert.equal(new Set(pointers).size, 1, 'all workers borrow exactly the same C handle');
    assert.equal(typeof pointers[0], 'bigint');
    const values = await Promise.all(pending);
    values.forEach((value, i) => near(value, fixture.cases[i].embedding));
    near(model.encodeText(fixture.cases[0].input), fixture.cases[0].embedding);
    assert.notEqual(values[0].buffer, values[1].buffer);
    values[0].fill(0);
    near(values[1], fixture.cases[1].embedding);
  } finally { model.close(); }
});

test('mixed PCM/file/text calls own snapshots and results', async () => {
  const model = new CLAP(modelPath);
  try {
    const samples = pcm();
    const expected = model.encodeAudio(samples);
    // A minimal PCM WAV exercises the file overload with no external decoder.
    const filename = path.join(directory, 'audio.wav');
    const wav = Buffer.alloc(44 + samples.length * 4);
    wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
    wav.writeUInt32LE(16, 16); wav.writeUInt16LE(3, 20); wav.writeUInt16LE(1, 22);
    wav.writeUInt32LE(48000, 24); wav.writeUInt32LE(192000, 28);
    wav.writeUInt16LE(4, 32); wav.writeUInt16LE(32, 34); wav.write('data', 36);
    wav.writeUInt32LE(samples.length * 4, 40);
    samples.forEach((value, i) => wav.writeFloatLE(value, 44 + i * 4));
    fs.writeFileSync(filename, wav);
    const tokens = tokenIds(fixture.cases[0]);
    const pending = [model.encodeAudioAsync(samples), model.encodeAudioAsync(filename, 0), model.encodeTextAsync(tokens)];
    samples.fill(NaN); tokens.fill(999999);
    const [audio, file, text] = await Promise.all(pending);
    near(audio, expected); near(file, expected); near(text, fixture.cases[0].embedding);
  } finally { model.close(); }
});

test('concurrent native errors keep their worker TLS diagnostic', async () => {
  const model = new CLAP(modelPath);
  try {
    const missing = path.join(directory, 'unique-missing-audio.wav');
    let tokenError;
    try { model.encodeText([2147483647]); } catch (error) { tokenError = error; }
    let fileError;
    try { model.encodeAudio(missing); } catch (error) { fileError = error; }
    assert.ok(tokenError); assert.ok(fileError);
    assert.notEqual(tokenError.message, fileError.message);
    const result = await Promise.allSettled([
      model.encodeTextAsync([2147483647]), model.encodeAudioAsync(missing),
      model.encodeTextAsync(fixture.cases[0].input),
    ]);
    for (const [index, expected] of [tokenError, fileError].entries()) {
      assert.equal(result[index].status, 'rejected');
      assert.equal(result[index].reason.name, expected.name);
      assert.equal(result[index].reason.message, expected.message);
    }
    assert.equal(result[2].status, 'fulfilled');
    near(result[2].value, fixture.cases[0].embedding);
  } finally { model.close(); }
});

test('close blocks new leases and defers one destruction through all inflight calls', async () => {
  const start = destroys;
  const model = new CLAP(modelPath);
  const pending = [model.encodeTextAsync('hello'), model.encodeAudioAsync(pcm())];
  model.close(); model.close();
  assert.equal(destroys, start);
  assert.throws(() => model.encodeText('hello'), /closed/);
  await assert.rejects(model.encodeTextAsync('hello'), /closed/);
  await assert.rejects(model.encodeAudioAsync(pcm()), /closed/);
  const result = await Promise.all(pending);
  assert.ok(result.every(value => value instanceof Float32Array && value.every(Number.isFinite)));
  assert.equal(destroys, start + 1);
  model.close();
  assert.equal(destroys, start + 1);
});

for (const mode of ['constructor', 'startup', 'empty', 'malformed']) {
  test(`worker ${mode} failure releases its lease`, async () => {
    const start = destroys;
    const model = new CLAP(modelPath);
    fault = mode;
    const pending = model.encodeTextAsync('hello');
    model.close();
    await assert.rejects(pending, mode === 'empty' ? /without.*result|exited/i : mode === 'malformed' ? /malformed/i : /injected worker/);
    assert.equal(destroys, start + 1);
  });
}

test('message delivery does not settle or release the native lease before worker exit', async () => {
  const start = destroys;
  const model = new CLAP(modelPath);
  fault = 'linger';
  const message = new Promise(resolve => { nextMessage = resolve; });
  let settled = false;
  const pending = model.encodeTextAsync('hello').finally(() => { settled = true; });
  model.close();
  await message;
  assert.equal(settled, false);
  assert.equal(destroys, start);
  await pending;
  assert.equal(destroys, start + 1);
});

test('sync calls lease the handle across reentrant close while async work is pending', async () => {
  const start = destroys;
  const model = new CLAP(modelPath);
  const pending = model.encodeTextAsync('world');
  const decode = koffi.decode;
  koffi.decode = function (...args) {
    model.close();
    assert.equal(destroys, start);
    return decode.apply(this, args);
  };
  try { near(model.encodeText('hello'), fixture.cases[0].embedding); }
  finally { koffi.decode = decode; model.close(); }
  await pending;
  assert.equal(destroys, start + 1);
});

test('GC finalizer releases an async owner exactly once without explicit close', () => {
  const child = spawnSync(process.execPath, ['--expose-gc', '-e', `
    const assert = require('node:assert/strict');
    const { createRequire } = require('node:module');
    const path = require('node:path');
    const local = createRequire(path.join(process.argv[1], 'index.js'));
    const koffi = local('koffi');
    const load = koffi.load;
    let destroyed = 0;
    koffi.load = (...args) => {
      const lib = load(...args);
      return { func(...signature) {
        const fn = lib.func(...signature);
        return signature[0] === 'clap_destroy' ? (...args) => { destroyed++; return fn(...args); } : fn;
      }};
    };
    const { CLAP } = require(process.argv[1]);
    (async () => {
      const pending = (() => { const owner = new CLAP(process.argv[2]); return owner.encodeTextAsync('hello'); })();
      const value = await pending;
      assert.ok(value instanceof Float32Array && value.every(Number.isFinite));
      for (let i = 0; i < 200 && destroyed === 0; i++) {
        global.gc(); await new Promise(resolve => setTimeout(resolve, 10));
      }
      assert.equal(destroyed, 1);
      for (let i = 0; i < 3; i++) { global.gc(); await new Promise(setImmediate); }
      assert.equal(destroyed, 1);
    })().catch(error => { console.error(error); process.exitCode = 1; });
  `, path.resolve(modulePath), modelPath], { encoding: 'utf8', timeout: 15000 });
  assert.equal(child.status, 0, child.stderr);
});

test('npm package includes both private worker helpers', () => {
  const args = ['pack', '--dry-run', '--json', '--ignore-scripts', '--cache', path.join(directory, 'npm-cache')];
  const npmCli = process.env.npm_execpath;
  const command = npmCli ? process.execPath : process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const packed = spawnSync(command, npmCli ? [npmCli, ...args] : args,
    { cwd: path.resolve(modulePath), encoding: 'utf8', timeout: 30000, shell: !npmCli && process.platform === 'win32' });
  assert.equal(packed.status, 0, packed.stderr);
  const files = JSON.parse(packed.stdout)[0].files.map(file => file.path);
  for (const filename of ['index.js', 'index.d.ts', '_native.js', '_encode-worker.js']) assert.ok(files.includes(filename), filename);
  assert.ok(files.every(filename => !filename.includes('node_modules') && !filename.endsWith('.so')));
});

test('async validation rejects Promises, brand-checks and closes safely on reentrant getters', async () => {
  const model = new CLAP(modelPath);
  const start = destroys;
  const input = [0, 2];
  Object.defineProperty(input, 0, { get() { model.close(); return 0; } });
  await assert.rejects(model.encodeTextAsync(input), /closed/);
  assert.equal(destroys, start + 1);
  await assert.rejects(model.encodeAudioAsync(new Float32Array([NaN])), /finite/);
  await assert.rejects(CLAP.prototype.encodeTextAsync.call({}, 'hello'), TypeError);
  await assert.rejects(CLAP.prototype.encodeAudioAsync.call({}, pcm()), TypeError);
});
