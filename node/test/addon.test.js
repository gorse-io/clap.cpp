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
  for (const x of [null, undefined, {}, 3, [], [-1], [0.5], [NaN], [Infinity], [2147483648],
    ['1'], new Float32Array([1]), new Int32Array([-1])]) assert.throws(() => m.encodeText(x));
  assert.throws(() => m.encodeText([999999]));
  for (const p of ['', 'a\0b', path.join(fixtures, 'missing.wav'), modelPath]) assert.throws(() => m.encodeAudio(p));
  m.close();
});
