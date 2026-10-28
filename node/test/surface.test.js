'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const binding = require(process.env.CLAP_CPP_MODULE || '..');

test('only CLAP and its encoding/lifecycle methods are public', () => {
  assert.deepEqual(Object.keys(binding), ['CLAP']);
  assert.equal(binding.CLAP.name, 'CLAP');
  assert.deepEqual(Object.getOwnPropertyNames(binding.CLAP.prototype).sort(),
    ['constructor', 'encodeAudio', 'encodeText', 'close'].sort());
});
