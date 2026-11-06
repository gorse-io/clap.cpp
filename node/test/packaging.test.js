'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.dirname(require.resolve(process.env.CLAP_CPP_MODULE || '..'));
const platforms = {
  'linux-x64': 'libclap.so',
  'linux-arm64': 'libclap.so',
  'darwin-arm64': 'libclap.dylib',
  'win32-x64': 'clap.dll',
};
const lifecycle = ['preinstall', 'install', 'postinstall', 'build', 'rebuild', 'prepare', 'prepack', 'postpack'];

function checkManifest(manifest) {
  assert.deepEqual(Object.keys(manifest.dependencies), ['koffi']);
  assert.equal(manifest.engines.node, '>=22.0.0');
  for (const hook of lifecycle) assert.equal(manifest.scripts[hook], undefined, `no ${hook} hook`);
  for (const name of ['scripts', 'native', 'native-tests', 'build']) {
    assert.ok(!manifest.files.some(p => p === name || p.startsWith(`${name}/`)), `no ${name} allowlist`);
  }
  assert.ok(manifest.files.includes('runtime'));
  assert.ok(manifest.files.includes('test/tiny_models.py'));
}

test('manifest has no native build lifecycle or native-source allowlists', () => {
  checkManifest(JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')));
  assert.ok(!fs.existsSync(path.join(root, 'scripts')), 'scripts directory is removed');
  assert.ok(!fs.existsSync(path.join(root, 'test/build.test.js')), 'build tests are removed');
});

test('real npm tarball contains prepared runtimes, licenses and tests and installs without a project build', { timeout: 180000 }, t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clap-package-test-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const packed = spawnSync('npm', ['pack', '--json', '--pack-destination', dir], {
    cwd: root, encoding: 'utf8', shell: process.platform === 'win32', maxBuffer: 10 * 1024 * 1024,
    timeout: 60000,
  });
  assert.equal(packed.error, undefined);
  assert.equal(packed.status, 0, packed.stderr);
  const archive = path.join(dir, JSON.parse(packed.stdout)[0].filename);
  // Read the actual archive, not just npm's dry-run file listing.
  const python = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
  const inspected = spawnSync(python, ['-c', `
import json, sys, tarfile
with tarfile.open(sys.argv[1]) as archive:
    files = [member.name.removeprefix('package/') for member in archive.getmembers() if member.isfile()]
    manifest = json.load(archive.extractfile('package/package.json'))
print(json.dumps(dict(files=files, manifest=manifest)))
`, archive], { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
  assert.equal(inspected.error, undefined);
  assert.equal(inspected.status, 0, inspected.stderr);
  const { files, manifest } = JSON.parse(inspected.stdout);
  checkManifest(manifest);
  for (const filename of ['index.js', 'index.d.ts', 'README.md', 'LICENSE', 'package.json',
    'test/generate-fixtures.py', 'test/tiny_models.py', 'test/types.ts', 'test/tsconfig.json']) {
    assert.ok(files.includes(filename), `missing ${filename}`);
  }
  for (const name of fs.readdirSync(path.join(root, 'test')).filter(n => n.endsWith('.test.js'))) {
    assert.ok(files.includes(`test/${name}`), `missing test/${name}`);
  }
  assert.ok(!files.some(p => /^(scripts|native|native-tests|build|src|include|cmake|ggml|thirdparty|node_modules)\//.test(p) ||
    p.endsWith('.node') || p.endsWith('.tgz') || p.includes('/.git/') || p.startsWith('test/fixtures/')),
  'no build scripts, native sources, dependencies or generated test artifacts');
  const host = `${process.platform}-${process.arch}`;
  const required = process.env.CLAP_CPP_REQUIRE_ALL_RUNTIMES === '1' ? Object.keys(platforms) : [host];
  assert.ok(platforms[host], `unsupported test host ${host}`);
  for (const target of required) {
    assert.ok(files.includes(`runtime/${target}/${platforms[target]}`), `missing ${target} runtime`);
  }
  // Validate licenses for every bundled platform, even in single-platform checkouts.
  for (const [target, library] of Object.entries(platforms)) {
    if (!files.some(p => p.startsWith(`runtime/${target}/`))) continue;
    assert.ok(files.includes(`runtime/${target}/${library}`), `missing ${target} library`);
    for (const name of ['ggml', 'icu', 'json', 'miniaudio']) {
      assert.ok(files.includes(`runtime/${target}/licenses/${name}.txt`), `missing ${target} ${name} license`);
    }
  }
  const installed = spawnSync('npm', ['install', '--prefix', path.join(dir, 'installed'), '--omit=dev',
    '--no-audit', '--no-fund', archive], {
    encoding: 'utf8', shell: process.platform === 'win32', timeout: 90000,
    maxBuffer: 10 * 1024 * 1024,
  });
  assert.equal(installed.error, undefined);
  assert.equal(installed.status, 0, installed.stderr);
  const pkg = path.join(dir, 'installed', 'node_modules', manifest.name);
  const smoke = spawnSync(process.execPath, ['-e', `
    const assert = require('node:assert/strict');
    const binding = require(${JSON.stringify(pkg)});
    assert.deepEqual(Object.keys(binding), ['CLAP']);
    assert.throws(() => new binding.CLAP('missing-model.gguf'), /.+/);
  `], { encoding: 'utf8', timeout: 30000 });
  assert.equal(smoke.error, undefined);
  assert.equal(smoke.status, 0, smoke.stderr);
  for (const filename of ['index.js', 'index.d.ts', 'test/generate-fixtures.py', 'test/tiny_models.py']) {
    assert.deepEqual(fs.readFileSync(path.join(pkg, filename)), fs.readFileSync(path.join(root, filename)));
  }
  const generated = spawnSync(python, ['test/generate-fixtures.py'], { cwd: pkg, encoding: 'utf8', timeout: 30000 });
  assert.equal(generated.error, undefined);
  assert.equal(generated.status, 0, generated.stderr);
  const tested = spawnSync(process.execPath, ['--test', 'test/addon.test.js', 'test/surface.test.js', 'test/ffi.test.js',
    'test/runtime.test.js'], { cwd: pkg, encoding: 'utf8', timeout: 30000, maxBuffer: 10 * 1024 * 1024,
    env: { ...process.env, CLAP_CPP_MODULE: pkg } });
  assert.equal(tested.error, undefined);
  assert.equal(tested.status, 0, tested.stdout + tested.stderr);
});
