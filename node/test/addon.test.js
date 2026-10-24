'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const { spawnSync } = require('node:child_process');
const root = process.env.CLAP_CPP_MODULE || '..';
const { Model, decodeAudio, resampleAudio } = require(root);
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
test('native exports and strict constructor validation', () => {
  assert.equal(typeof Model, 'function');
  for (const p of [undefined, null, 1, '', 'a\0b']) assert.throws(() => new Model(p));
  for (const n of [-1, 0.5, NaN, Infinity, 2147483648, '1', null]) assert.throws(() => new Model(modelPath, n));
  assert.throws(() => Model(modelPath));
  assert.throws(() => new Model(path.join(fixtures, 'missing.gguf')));
});
test('real CLAP tokenization and text inference, with owned outputs', () => {
  const m = new Model(modelPath, 1);
  assert.equal(m.projectionDimension, 3);
  assert.equal(m.audioSampleRate, 48000);
  assert.throws(() => { m.projectionDimension = 5; }, TypeError);
  const ids = m.tokenize('hello', 32);
  assert.ok(ids instanceof Int32Array);
  assert.ok(ids.length > 2);
  assert.equal(ids[0], 0);
  assert.equal(ids.at(-1), 2);
  assert.ok(m.tokenize('你好 🎵', 32).length > 2);
  const a = m.text('hello');
  vector(a, 3);
  near(a, m.text(ids));
  near(a, m.text(Array.from(ids)));
  assert.ok(Math.abs(Math.hypot(...a) - 1) < 1e-5);
  const oracle = path.join(fixtures, 'reference.json');
  if (fs.existsSync(oracle)) {
    const ref = JSON.parse(fs.readFileSync(oracle, 'utf8'));
    near(m.text(ref.ids), ref.values);
  }
  const b = m.text('hello');
  a.fill(99);
  near(b, m.text('hello'));
  m.close(); m.close();
  vector(b, 3);
  for (const f of [() => m.text('x'), () => m.tokenize('x'), () => m.audio(pcm),
    () => m.audioFrames(pcm), () => m.clamp3Features(pcm), () => m.projectionDimension,
    () => m.audioSampleRate]) assert.throws(f, /closed/i);
});
test('text rejects a model closed by a reentrant token array getter', () => {
  const child = spawnSync(process.execPath, ['-e', `
    const assert = require('node:assert/strict');
    const { Model } = require(${JSON.stringify(require.resolve(root))});
    const m = new Model(${JSON.stringify(modelPath)}, 1);
    const ids = [0, 2];
    let calls = 0;
    Object.defineProperty(ids, 0, { get() { ++calls; m.close(); return 0; } });
    assert.throws(() => m.text(ids), /closed/i);
    assert.equal(calls, 1);
    assert.throws(() => m.text('hello'), /closed/i);
    m.close();
    console.log('reentrant close rejected safely');
  `], { encoding: 'utf8', timeout: 30000 });
  assert.equal(child.error, undefined);
  assert.equal(child.signal, null, `child signal: ${child.signal}\n${child.stderr}`);
  assert.equal(child.status, 0, child.stderr);
  assert.match(child.stdout, /reentrant close rejected safely/);
});
test('CLAP accepts actual waveforms and rejects unsupported frame APIs', () => {
  const m = new Model(modelPath, 1);
  vector(m.audio(pcm, 24000), 3);
  vector(m.audio(pcm, 24000, 0), 3);
  assert.throws(() => m.audioFrames(pcm, 24000));
  assert.throws(() => m.clamp3Features(pcm, 24000));
  m.close();
});
test('MERT actual frame, pooled and CLaMP3-feature inference', () => {
  const m = new Model(mertPath, 1);
  assert.equal(m.audioSampleRate, 24000);
  assert.equal(m.projectionDimension, 4);
  const f = m.audioFrames(pcm, 24000);
  assert.deepEqual(Object.keys(f).sort(), ['dimension', 'frames', 'values']);
  assert.equal(f.dimension, 4);
  assert.ok(f.frames > 1);
  vector(f.values, f.frames * f.dimension);
  const pooled = m.audio(pcm, 24000);
  vector(pooled, 4);
  near(pooled, Array.from({ length: 4 }, (_, d) => {
    let sum = 0; for (let t = 0; t < f.frames; ++t) sum += f.values[t * 4 + d];
    return sum / f.frames;
  }));
  const c = m.clamp3Features(pcm, 24000);
  assert.equal(c.frames, 1); assert.equal(c.dimension, 4); vector(c.values, 4);
  assert.throws(() => m.tokenize('hello'));
  assert.throws(() => m.text('hello'));
  m.close(); vector(f.values, f.frames * 4);
});
test('strict method input validation and native exceptions', () => {
  const m = new Model(modelPath, 1);
  for (const x of [undefined, null, {}, [], new Uint8Array([1]), new Float32Array(),
    new Float32Array([NaN]), new Float32Array([Infinity])]) {
    assert.throws(() => m.audio(x)); assert.throws(() => m.audioFrames(x));
    assert.throws(() => m.clamp3Features(x));
  }
  for (const rate of [0, -1, 0.5, NaN, Infinity, 2147483648, '24000', null]) {
    assert.throws(() => m.audio(pcm, rate));
    assert.throws(() => m.audioFrames(pcm, rate));
    assert.throws(() => m.clamp3Features(pcm, rate));
  }
  for (const offset of [-1, 0.5, Infinity, NaN, 2 ** 53, '0', null]) assert.throws(() => m.audio(pcm, 24000, offset));
  for (const x of [null, undefined, {}, 3, [], [-1], [0.5], [NaN], [Infinity], [2147483648],
    ['1'], new Float32Array([1]), new Int32Array([-1])]) assert.throws(() => m.text(x));
  assert.throws(() => m.text([999999]));
  for (const x of [null, 1, undefined]) assert.throws(() => m.tokenize(x));
  for (const n of [0, -1, 0.5, NaN, Infinity, 2 ** 53, '1', null]) assert.throws(() => m.tokenize('x', n));
  m.close();
});
test('decode mono/stereo WAV, resample and retain owned PCM', () => {
  const a = decodeAudio(path.join(fixtures, 'stereo.wav'));
  assert.equal(a.sampleRate, 24000);
  vector(a.samples, 24000);
  assert.ok(Math.abs(a.samples[0] - 0.25) < 1e-4);
  const same = resampleAudio(a.samples, 24000, 24000);
  assert.deepEqual(same, a.samples);
  same[0] = 99; assert.notEqual(a.samples[0], 99);
  const b = resampleAudio(pcm, 24000, 48000);
  vector(b, 48000);
  vector(resampleAudio(pcm, 24000), 48000);
  vector(resampleAudio(pcm, 24000, 16000), 16000);
  assert.equal(resampleAudio(new Float32Array([0.5]), 24000, 48000).length, 2);
});
test('audio utility strict validation and decoder error translation', () => {
  for (const p of [undefined, null, 1, '', 'a\0b', path.join(fixtures, 'missing.wav'), modelPath]) assert.throws(() => decodeAudio(p));
  for (const x of [null, [], new Float64Array([1]), new Float32Array(), new Float32Array([NaN])]) assert.throws(() => resampleAudio(x, 24000));
  for (const r of [undefined, null, 0, -1, 0.5, Infinity, NaN, 2147483648, '24000']) assert.throws(() => resampleAudio(pcm, r));
  for (const r of [null, 0, -1, 0.5, Infinity, NaN, 2147483648, '24000']) assert.throws(() => resampleAudio(pcm, 24000, r));
});
