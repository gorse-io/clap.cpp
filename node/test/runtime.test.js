'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const entry = require.resolve(process.env.CLAP_CPP_MODULE || '..');
const platforms = {
  'linux-x64': 'libclap.so',
  'linux-arm64': 'libclap.so',
  'darwin-arm64': 'libclap.dylib',
  'win32-x64': 'clap.dll',
};

function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clap-runtime-test-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  fs.copyFileSync(entry, path.join(dir, 'index.js'));
  return dir;
}
function load(dir, platform, arch) {
  // Stop at the FFI load boundary: these tests select paths, not native ABI.
  return spawnSync(process.execPath, ['-e', `
    Object.defineProperty(process, 'platform', { value: ${JSON.stringify(platform)} });
    Object.defineProperty(process, 'arch', { value: ${JSON.stringify(arch)} });
    const Module = require('node:module');
    const original = Module._load;
    Module._load = function(name, ...args) {
      if (name === 'koffi') return { load(filename) {
        console.log(filename);
        throw new Error('selected runtime');
      } };
      return original.call(this, name, ...args);
    };
    require(${JSON.stringify(path.join(dir, 'index.js'))});
  `], { encoding: 'utf8' });
}

for (const [target, library] of Object.entries(platforms)) {
  test(`selects only the package-local ${target} runtime`, t => {
    const dir = fixture(t);
    const [platform, arch] = target.split('-');
    const runtime = path.join(dir, 'runtime', target, library);
    fs.mkdirSync(path.dirname(runtime), { recursive: true });
    fs.writeFileSync(runtime, 'selection test only');
    const result = load(dir, platform, arch);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /selected runtime/);
    assert.equal(result.stdout.trim(), runtime);
  });
}

test('rejects unsupported operating systems and architectures before loading', t => {
  const dir = fixture(t);
  for (const [platform, arch] of [['freebsd', 'x64'], ['linux', 'ia32'], ['darwin', 'x64'], ['win32', 'arm64']]) {
    const result = load(dir, platform, arch);
    assert.equal(result.status, 1);
    assert.match(result.stderr, new RegExp(`Unsupported platform or architecture: ${platform}-${arch}`));
    assert.equal(result.stdout, '');
  }
});

test('missing runtime reports its exact package-local path without falling back', t => {
  const dir = fixture(t);
  // An old-layout or another architecture's binary must not be selected.
  for (const relative of ['build/runtime/libclap.so', 'runtime/linux-arm64/libclap.so']) {
    const filename = path.join(dir, relative);
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, 'not the host runtime');
  }
  const result = load(dir, 'linux', 'x64');
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Missing prebuilt clap runtime for linux-x64/);
  assert.ok(result.stderr.includes(path.join(dir, 'runtime', 'linux-x64', 'libclap.so')));
  assert.equal(result.stdout, '');
});
