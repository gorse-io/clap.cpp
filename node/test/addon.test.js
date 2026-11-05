'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const { spawnSync } = require('node:child_process');
const root = process.env.CLAP_CPP_MODULE || '..';
const { CLAP } = require(root);
const fixtures = path.join(__dirname, 'fixtures');
const modelPath = path.join(fixtures, 'clap.gguf');
const mertPath = path.join(fixtures, 'mert.gguf');
function vector(v, n) {
  assert.ok(v instanceof Float32Array);
  assert.equal(v.length, n);
  assert.ok(v.every(Number.isFinite));
  assert.ok(v.some(x => x !== 0));
}
function near(a, b, tolerance = 2e-5) {
  assert.equal(a.length, b.length);
  a.forEach((x, i) => assert.ok(Math.abs(x - b[i]) < tolerance, `${i}: ${x} != ${b[i]}`));
}
const pcm = Float32Array.from({ length: 24000 }, (_, i) => 0.2 * Math.sin(i * 0.031));
test('strict constructor validation', () => {
  assert.equal(typeof CLAP, 'function');
  for (const p of [undefined, null, 1, '', 'a\0b']) assert.throws(() => new CLAP(p));
  for (const n of [-1, 0.5, NaN, Infinity, 2147483648, '1', null]) assert.throws(() => new CLAP(modelPath, n));
  assert.throws(() => CLAP(modelPath));
  assert.throws(() => new CLAP(path.join(fixtures, 'missing.gguf')));
});
test('real CLAP text inference with owned outputs and scalar oracle', () => {
  const m = new CLAP(modelPath, 1);
  const a = m.encodeText('hello');
  vector(a, 3);
  const oracle = path.join(fixtures, 'reference.json');
  if (fs.existsSync(oracle)) {
    const ref = JSON.parse(fs.readFileSync(oracle, 'utf8'));
    near(a, m.encodeText(ref.ids));
    near(a, m.encodeText(Int32Array.from(ref.ids)));
    near(a, ref.values);
  }
  vector(m.encodeText('你好 🎵'), 3);
  assert.ok(Math.abs(Math.hypot(...a) - 1) < 1e-5);
  const b = m.encodeText('hello');
  a.fill(99);
  near(b, m.encodeText('hello'));
  m.close(); m.close();
  vector(b, 3);
  for (const f of [() => m.encodeText('x'), () => m.encodeAudio(pcm),
    () => m.encodeAudio(path.join(fixtures, 'stereo.wav'))]) assert.throws(f, /closed/i);
});
test('encodeText rejects a model closed by a reentrant token array getter', () => {
  const child = spawnSync(process.execPath, ['-e', `
    const assert = require('node:assert/strict');
    const { CLAP } = require(${JSON.stringify(require.resolve(root))});
    const m = new CLAP(${JSON.stringify(modelPath)}, 1);
    const ids = [0, 2];
    let calls = 0;
    Object.defineProperty(ids, 0, { get() { ++calls; m.close(); return 0; } });
    assert.throws(() => m.encodeText(ids), /closed/i);
    assert.equal(calls, 1);
    assert.throws(() => m.encodeText('hello'), /closed/i);
    m.close();
    console.log('reentrant close rejected safely');
  `], { encoding: 'utf8', timeout: 30000 });
  assert.equal(child.error, undefined);
  assert.equal(child.signal, null, `child signal: ${child.signal}\n${child.stderr}`);
  assert.equal(child.status, 0, child.stderr);
  assert.match(child.stdout, /reentrant close rejected safely/);
});
test('token Proxy length and index traps finish before borrowing a handle', () => {
  for (const key of ['length', '0']) {
    const m = new CLAP(modelPath, 1);
    const ids = new Proxy([0, 2], { get(target, property, receiver) {
      if (property === key) m.close();
      return Reflect.get(target, property, receiver);
    } });
    assert.throws(() => m.encodeText(ids), /closed/i);
    m.close();
  }
});
test('typed array conversion ignores user getters, iterators and species', () => {
  const m = new CLAP(modelPath, 1);
  try {
    const ids = new Int32Array([0, 2]);
    const waveform = new Float32Array(pcm);
    for (const value of [ids, waveform]) {
      for (const key of ['length', 'buffer', 'byteOffset', 'constructor', Symbol.iterator]) {
        Object.defineProperty(value, key, { get() { throw new Error('must not run a user getter'); } });
      }
    }
    vector(m.encodeText(ids), 3);
    vector(m.encodeAudio(waveform, 24000), 3);
    assert.throws(() => m.encodeText(new Proxy(ids, {})), TypeError);
    assert.throws(() => m.encodeAudio(new Proxy(waveform, {})), TypeError);
  } finally { m.close(); }
});
test('instances cannot be copied or impersonated to borrow native ownership', () => {
  const m = new CLAP(modelPath, 1);
  try {
    assert.deepEqual(Reflect.ownKeys(m), []);
    for (const receiver of [{ ...m }, Object.create(m), Object.create(CLAP.prototype),
      new Proxy(m, {}), {}, null]) {
      for (const [method, args] of [['close', []], ['encodeText', ['hello']], ['encodeAudio', [pcm]]]) {
        assert.throws(() => Reflect.apply(CLAP.prototype[method], receiver, args), TypeError);
      }
    }
    vector(m.encodeText('hello'), 3);
  } finally { m.close(); }
});
test('explicit close and garbage collection keep separate owners and returned vectors safe', () => {
  const entry = require.resolve(root);
  const koffiPath = require.resolve('koffi', { paths: [path.dirname(entry)] });
  const child = spawnSync(process.execPath, ['--expose-gc', '-e', `
    const assert = require('node:assert/strict');
    const koffi = require(${JSON.stringify(koffiPath)});
    const load = koffi.load;
    let destroys = 0;
    koffi.load = (...args) => {
      const lib = load(...args);
      return { func(...declaration) {
        const fn = lib.func(...declaration);
        if (declaration[0] !== 'clap_destroy') return fn;
        return pointer => { ++destroys; return fn(pointer); };
      } };
    };
    const { CLAP } = require(${JSON.stringify(entry)});
    (async () => {
      const survivor = new CLAP(${JSON.stringify(modelPath)}, 1);
      const vectors = [];
      const weak = [];
      for (let i = 0; i < 12; ++i) {
        let model = new CLAP(${JSON.stringify(modelPath)}, 1);
        vectors.push(model.encodeText('hello'));
        weak.push(new WeakRef(model));
        if (i % 2 === 0) { model.close(); model.close(); }
        model = null;
      }
      assert.equal(destroys, 6, 'close destroys each explicitly closed owner exactly once');
      let collected = false;
      for (let i = 0; i < 100; ++i) {
        await new Promise(resolve => setImmediate(resolve));
        global.gc();
        await new Promise(resolve => setImmediate(resolve));
        if (weak.every(ref => ref.deref() === undefined)) { collected = true; break; }
      }
      assert.ok(collected, 'models should be collectible (registry must not retain owners)');
      global.gc();
      for (let i = 0; i < 100 && destroys !== 12; ++i) {
        await new Promise(resolve => setImmediate(resolve));
      }
      assert.equal(destroys, 12, 'GC destroys only the remaining unclosed owners');
      assert.deepEqual(survivor.encodeText('hello'), vectors[0]);
      survivor.close(); survivor.close();
      assert.equal(destroys, 13);
      vectors.forEach(v => assert.ok(v instanceof Float32Array && v.every(Number.isFinite)));
      console.log('GC and close owners remain independent');
    })().catch(error => { console.error(error); process.exitCode = 1; });
  `], { encoding: 'utf8', timeout: 60000 });
  assert.equal(child.error, undefined);
  assert.equal(child.signal, null, child.stderr);
  assert.equal(child.status, 0, child.stderr);
  assert.match(child.stdout, /GC and close owners remain independent/);
});
test('CLAP encodes waveforms and audio paths with automatic resampling', () => {
  const m = new CLAP(modelPath, 1);
  vector(m.encodeAudio(pcm, 24000), 3);
  near(m.encodeAudio(pcm, 24000), m.encodeAudio(pcm, 24000, 0));
  const filename = path.join(fixtures, 'stereo.wav');
  const expected = m.encodeAudio(new Float32Array(24000).fill(0.25), 24000);
  near(m.encodeAudio(filename), expected);
  near(m.encodeAudio(filename, 0), expected);
  m.close();
});
test('MERT exposes pooled audio through encodeAudio', () => {
  const m = new CLAP(mertPath, 1);
  vector(m.encodeAudio(pcm, 24000), 4);
  assert.throws(() => m.encodeText('hello'));
  m.close();
});
test('strict method input validation and native exceptions', () => {
  const m = new CLAP(modelPath, 1);
  for (const x of [undefined, null, {}, [], new Uint8Array([1]), new Float32Array(),
    new Float32Array([NaN]), new Float32Array([Infinity])]) assert.throws(() => m.encodeAudio(x));
  for (const rate of [0, -1, 0.5, NaN, Infinity, 2147483648, '24000', null]) {
    assert.throws(() => m.encodeAudio(pcm, rate));
  }
  for (const offset of [-1, 0.5, Infinity, NaN, 2 ** 53, '0', null]) {
    assert.throws(() => m.encodeAudio(pcm, 24000, offset));
    assert.throws(() => m.encodeAudio(path.join(fixtures, 'stereo.wav'), offset));
  }
  for (const x of [null, undefined, {}, 3, 'a\0b', [], [-1], [0.5], [NaN], [Infinity], [2147483648],
    ['1'], new Float32Array([1]), new Int32Array([-1])]) assert.throws(() => m.encodeText(x));
  assert.throws(() => m.encodeText([999999]));
  for (const p of ['', 'a\0b', path.join(fixtures, 'missing.wav'), modelPath]) assert.throws(() => m.encodeAudio(p));
  m.close();
});

test('constructor options preserve text loading and do not ignore mmproj', () => {
  for (const options of [{}, { threads: 1 }, { threads: 1, mmproj: undefined }]) {
    const m = new CLAP(modelPath, options);
    try { vector(m.encodeText('hello'), 3); } finally { m.close(); }
  }
  assert.throws(() => new CLAP(path.join(fixtures, 'embeddinggemma2-dispatch.gguf'), {
    threads: 1, mmproj: path.join(fixtures, 'missing-audio-mmproj.gguf')
  }), /missing-audio-mmproj/);
  assert.throws(() => new CLAP(modelPath, { mmproj: 'missing-audio-mmproj.gguf' }),
    error => error instanceof Error && !(error instanceof TypeError));
  for (const mmproj of [null, 1, '', 'a\0b']) {
    assert.throws(() => new CLAP(modelPath, { mmproj }), TypeError);
  }
  for (const threads of [-1, 0.5, NaN, null, '1']) {
    assert.throws(() => new CLAP(modelPath, { threads }), TypeError);
  }
});
