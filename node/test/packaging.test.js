'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.join(__dirname, '..');
test('prepack vendors the root build graph, pinned sources, licenses and fixture generator', () => {
  const r = spawnSync(process.execPath, ['scripts/vendor.js'], { cwd: root, encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  for (const p of ['native/CMakeLists.txt', 'native/LICENSE', 'native/include/clap.h', 'native/src/model.cpp', 'native/cmake/icu.cmake',
    'native/ggml/CMakeLists.txt', 'native/thirdparty/json/CMakeLists.txt',
    'native/thirdparty/icu/icu4c/source/common/unicode/uvernum.h',
    'native/thirdparty/miniaudio/miniaudio.h', 'native-tests/tiny_models.py',
    'LICENSE', 'THIRD_PARTY_LICENSES/ggml.txt', 'THIRD_PARTY_LICENSES/icu.txt',
    'THIRD_PARTY_LICENSES/miniaudio.txt', 'THIRD_PARTY_LICENSES/json.txt'])
    assert.ok(fs.existsSync(path.join(root, p)), p);
  for (const p of ['native/ggml/.git', 'native/thirdparty/icu/icu4c/source/data',
    'native/thirdparty/miniaudio/data', 'native/thirdparty/miniaudio/tests',
    'native/thirdparty/json/docs', 'native/thirdparty/json/tests', 'native/ggml/tests'])
    assert.ok(!fs.existsSync(path.join(root, p)), `not shipping ${p}`);
  const manifest = require('../package.json');
  assert.deepEqual(Object.keys(manifest.dependencies), ['koffi']);
  assert.equal(manifest.engines.node, '>=22.0.0');
  assert.ok(!fs.existsSync(path.join(root, 'CMakeLists.txt')));
  assert.ok(!fs.existsSync(path.join(root, 'bindings.cpp')));
  const packed = spawnSync('npm', ['pack', '--dry-run', '--ignore-scripts', '--json'], {
    cwd: root, encoding: 'utf8', shell: process.platform === 'win32', maxBuffer: 10 * 1024 * 1024,
  });
  assert.equal(packed.status, 0, packed.stderr);
  const files = JSON.parse(packed.stdout)[0].files.map(f => f.path);
  assert.ok(files.includes('native/CMakeLists.txt'));
  for (const name of fs.readdirSync(path.join(root, 'test')).filter(n => n.endsWith('.test.js')))
    assert.ok(files.includes(`test/${name}`), `missing test/${name}`);
  assert.ok(!files.some(p => p.startsWith('build/') || p.endsWith('.node') || p.includes('/.git/')));
});

function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clap-build-test-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  fs.mkdirSync(path.join(dir, 'scripts'));
  fs.mkdirSync(path.join(dir, 'native'));
  fs.copyFileSync(path.join(root, 'scripts/build.js'), path.join(dir, 'scripts/build.js'));
  fs.writeFileSync(path.join(dir, 'native/CMakeLists.txt'), '# test fixture');
  return dir;
}
test('failed compiler/configure is an installation failure; rebuild only removes owned outputs', t => {
  const dir = fixture(t);
  for (const name of ['cmake', 'runtime', 'keep']) {
    fs.mkdirSync(path.join(dir, 'build', name), { recursive: true });
    fs.writeFileSync(path.join(dir, 'build', name, 'sentinel'), 'preserve unrelated output');
  }
  const preload = path.join(dir, 'fail-cmake.cjs');
  fs.writeFileSync(preload, `require('node:child_process').spawnSync = () => ({ status: 37 });`);
  const r = spawnSync(process.execPath, ['--require', preload, 'scripts/build.js', '--rebuild'], { cwd: dir, encoding: 'utf8' });
  assert.equal(r.status, 1);
  assert.match(r.stderr, /CMake failed \(37\)/);
  assert.ok(fs.existsSync(path.join(dir, 'build/keep/sentinel')));
  assert.ok(!fs.existsSync(path.join(dir, 'build/cmake')));
  assert.ok(!fs.existsSync(path.join(dir, 'build/runtime')));
});
test('rebuild refuses symlink outputs, including dangling symlinks', { skip: process.platform === 'win32' }, t => {
  for (const dangling of [false, true]) {
    const dir = fixture(t);
    const target = path.join(dir, dangling ? 'missing' : 'untouched');
    if (!dangling) fs.mkdirSync(target);
    fs.symlinkSync(target, path.join(dir, 'build'), 'dir');
    const r = spawnSync(process.execPath, ['scripts/build.js', '--rebuild'], { cwd: dir, encoding: 'utf8' });
    assert.equal(r.status, 1);
    assert.match(r.stderr, /Refusing unsafe output/);
    assert.ok(fs.lstatSync(path.join(dir, 'build')).isSymbolicLink());
    if (!dangling) assert.ok(fs.existsSync(target));
  }
});
