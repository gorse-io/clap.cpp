'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.join(__dirname, '..');
test('prepack vendors standalone native sources, licenses and fixture generator', () => {
  const r = spawnSync(process.execPath, ['scripts/vendor.js'], { cwd: root, encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  for (const p of ['native/include/clap.h', 'native/src/model.cpp', 'native/cmake/icu.cmake',
    'native/ggml/CMakeLists.txt', 'native/thirdparty/json/CMakeLists.txt',
    'native/thirdparty/icu/icu4c/source/common/unicode/uvernum.h',
    'native/thirdparty/miniaudio/miniaudio.h', 'native-tests/tiny_models.py',
    'LICENSE', 'THIRD_PARTY_LICENSES/ggml.txt', 'THIRD_PARTY_LICENSES/icu.txt',
    'THIRD_PARTY_LICENSES/miniaudio.txt', 'THIRD_PARTY_LICENSES/json.txt'])
    assert.ok(fs.existsSync(path.join(root, p)), p);
  assert.ok(!fs.existsSync(path.join(root, 'native/ggml/.git')));
  assert.ok(!fs.existsSync(path.join(root, 'native/thirdparty/icu/icu4c/source/data')));
  for (const p of ['native/thirdparty/miniaudio/data', 'native/thirdparty/miniaudio/tests',
    'native/thirdparty/json/docs', 'native/thirdparty/json/tests', 'native/ggml/tests'])
    assert.ok(!fs.existsSync(path.join(root, p)), `not shipping ${p}`);
});
