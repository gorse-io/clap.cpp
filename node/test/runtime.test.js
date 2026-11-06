'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const entry = require.resolve(process.env.CLAP_CPP_MODULE || '..');
const manifest = require(path.join(path.dirname(entry), 'package.json'));
const platforms = {
  'linux-x64': 'libclap.so',
  'linux-arm64': 'libclap.so',
  'darwin-arm64': 'libclap.dylib',
  'win32-x64': 'clap.dll',
};

function fixture(t, version = manifest.version) {
  const dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'clap-runtime-test-')));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  fs.copyFileSync(entry, path.join(dir, 'index.js'));
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify(manifest));
  fs.mkdirSync(path.join(dir, 'native'));
  fs.writeFileSync(path.join(dir, 'native/package.json'), JSON.stringify({ version }));
  return dir;
}
function load(dir, target, { missing = false, binary = true } = {}) {
  const [platform, arch] = target.split('-');
  const runtime = path.join(dir, 'native/runtime', platforms[target] || 'unused');
  // Mock only resolution/existence at the selection boundary. No fake binary is
  // created or loaded; real ABI and inference are exercised by packaging tests.
  return spawnSync(process.execPath, ['-e', `
    Object.defineProperty(process, 'platform', { value: ${JSON.stringify(platform)} });
    Object.defineProperty(process, 'arch', { value: ${JSON.stringify(arch)} });
    const Module = require('node:module');
    const resolve = Module._resolveFilename;
    Module._resolveFilename = function(name, ...args) {
      if (name.startsWith('@gorse/clap-cpp-')) {
        if (name !== ${JSON.stringify(`@gorse/clap-cpp-${target}/package.json`)}) throw Error('wrong optional package');
        if (${missing}) { const e = Error('not installed'); e.code = 'MODULE_NOT_FOUND'; throw e; }
        return ${JSON.stringify(path.join(dir, 'native/package.json'))};
      }
      return resolve.call(this, name, ...args);
    };
    const fs = require('node:fs');
    const exists = fs.existsSync;
    fs.existsSync = filename => filename === ${JSON.stringify(runtime)} ? ${binary} : exists(filename);
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

for (const target of Object.keys(platforms)) {
  test(`selects only the ${target} optional package runtime`, t => {
    const dir = fixture(t);
    const result = load(dir, target);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /selected runtime/);
    assert.equal(result.stdout.trim(), path.join(dir, 'native/runtime', platforms[target]));
  });
}

test('rejects unsupported operating systems and architectures before resolution', t => {
  const dir = fixture(t);
  for (const target of ['freebsd-x64', 'linux-ia32', 'darwin-x64', 'win32-arm64']) {
    const result = load(dir, target);
    assert.equal(result.status, 1);
    assert.ok(result.stderr.includes(`Unsupported platform or architecture: ${target}`));
    assert.equal(result.stdout, '');
  }
});

test('missing optional package gives installation guidance without a legacy fallback', t => {
  const dir = fixture(t);
  // Even a real library in the old layout must not be used.
  const result = load(dir, 'linux-x64', { missing: true });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Missing optional native package @gorse\/clap-cpp-linux-x64/);
  assert.match(result.stderr, /optional dependencies/);
  assert.equal(result.stdout, '');
});

test('rejects a native package with a different version before loading', t => {
  const dir = fixture(t, '0.0.0-mismatched');
  const result = load(dir, 'linux-x64');
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Native package version mismatch/);
  assert.ok(result.stderr.includes(manifest.version));
  assert.match(result.stderr, /0\.0\.0-mismatched/);
  assert.equal(result.stdout, '');
});

test('missing native library reports its exact path without falling back', t => {
  const dir = fixture(t);
  const result = load(dir, 'linux-x64', { binary: false });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Missing prebuilt clap runtime/);
  assert.ok(result.stderr.includes(path.join(dir, 'native/runtime/libclap.so')));
  assert.equal(result.stdout, '');
});
