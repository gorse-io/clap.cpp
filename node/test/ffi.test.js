'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

test('package entry is a JavaScript Koffi wrapper, not a project addon', () => {
  const filename = require.resolve(process.env.CLAP_CPP_MODULE || '..');
  const entry = fs.readFileSync(filename, 'utf8');
  assert.match(entry, /require\(['"]koffi['"]\)/);
  assert.doesNotMatch(entry, /clap_node\.node/);
  assert.equal(fs.existsSync(path.join(path.dirname(filename), 'bindings.cpp')), false);
});

test('real native buffers are freed when JavaScript output decoding throws', () => {
  const entry = require.resolve(process.env.CLAP_CPP_MODULE || '..');
  const koffiPath = require.resolve('koffi', { paths: [path.dirname(entry)] });
  const model = path.join(__dirname, 'fixtures', 'clap.gguf');
  const child = spawnSync(process.execPath, ['-e', `
    const assert = require('node:assert/strict');
    const koffi = require(${JSON.stringify(koffiPath)});
    const load = koffi.load;
    let frees = 0;
    // Observe the actual C deallocator, without substituting native inference.
    koffi.load = (...args) => {
      const lib = load(...args);
      return { func(...declaration) {
        const fn = lib.func(...declaration);
        if (declaration[0] !== 'clap_free') return fn;
        return pointer => { assert.notEqual(pointer, null); ++frees; return fn(pointer); };
      } };
    };
    const { CLAP } = require(${JSON.stringify(entry)});
    const m = new CLAP(${JSON.stringify(model)}, 1);
    const decode = koffi.decode;
    const failure = new RangeError('injected JS allocation failure');
    koffi.decode = () => { throw failure; };
    assert.throws(() => m.encodeText('hello'), error => error === failure);
    assert.equal(frees, 1);
    koffi.decode = decode;
    const result = m.encodeText('hello');
    assert.ok(result instanceof Float32Array);
    assert.equal(result.length, 3);
    assert.equal(frees, 2);
    m.close(); m.close();
    assert.ok(result.every(Number.isFinite));
    console.log('real buffers freed on both decode failure and success');
  `], { encoding: 'utf8', timeout: 120000 });
  assert.equal(child.error, undefined);
  assert.equal(child.signal, null, child.stderr);
  assert.equal(child.status, 0, child.stderr);
  assert.match(child.stdout, /real buffers freed/);
});
