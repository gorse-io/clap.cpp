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
const host = `${process.platform}-${process.arch}`;
const python = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const lifecycle = ['preinstall', 'install', 'postinstall', 'build', 'rebuild', 'prepare', 'prepack', 'postpack'];

function checkManifest(manifest) {
  assert.equal(manifest.name, 'clap-cpp');
  assert.deepEqual(Object.keys(manifest.dependencies), ['koffi']);
  assert.equal(manifest.engines.node, '>=22.0.0');
  assert.deepEqual(manifest.optionalDependencies, Object.fromEntries(
    Object.keys(platforms).map(target => [`@gorse/clap-cpp-${target}`, manifest.version])));
  for (const hook of lifecycle) assert.equal(manifest.scripts[hook], undefined, `no ${hook} hook`);
  for (const name of ['runtime', 'scripts', 'native', 'native-tests', 'build']) {
    assert.ok(!manifest.files.some(p => p === name || p.startsWith(`${name}/`)), `no ${name} allowlist`);
  }
  assert.ok(manifest.files.includes('test/tiny_models.py'));
}
function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: 'utf8', timeout: 180000, maxBuffer: 10 * 1024 * 1024,
    shell: command === 'npm' && process.platform === 'win32', ...options,
  });
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  return result.stdout;
}
function pack(source, destination) {
  const output = JSON.parse(run('npm', ['pack', '--json', '--pack-destination', destination], { cwd: source }));
  const archive = path.join(destination, output[0].filename);
  // Inspect the real archive, not an npm dry-run file list.
  const inspected = JSON.parse(run(python, ['-c', `
import json, sys, tarfile
with tarfile.open(sys.argv[1]) as archive:
    files = [m.name.removeprefix('package/') for m in archive.getmembers() if m.isfile()]
    manifest = json.load(archive.extractfile('package/package.json'))
print(json.dumps(dict(files=files, manifest=manifest)))
`, archive]));
  return { archive, ...inspected };
}
function checkNative(packed, target, version) {
  const { files, manifest } = packed;
  const [osName, cpu] = target.split('-');
  assert.equal(manifest.name, `@gorse/clap-cpp-${target}`);
  assert.equal(manifest.version, version);
  assert.deepEqual(manifest.os, [osName]);
  assert.deepEqual(manifest.cpu, [cpu]);
  assert.deepEqual(manifest.libc, osName === 'linux' ? ['glibc'] : undefined);
  for (const hook of lifecycle) assert.equal(manifest.scripts?.[hook], undefined);
  const expected = ['package.json', 'README.md', 'LICENSE', `runtime/${platforms[target]}`,
    ...['clap', 'ggml', 'icu', 'json', 'miniaudio'].map(name => `runtime/licenses/${name}.txt`)];
  assert.deepEqual([...files].sort(), expected.sort(), `native package contents for ${target}`);
}

test('manifest declares exact-version optional packages and no native build hooks', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  checkManifest(manifest);
  const lock = path.join(root, 'package-lock.json');
  if (fs.existsSync(lock)) {
    assert.deepEqual(JSON.parse(fs.readFileSync(lock, 'utf8')).packages[''].optionalDependencies,
      manifest.optionalDependencies);
  }
  assert.ok(!fs.existsSync(path.join(root, 'scripts')), 'scripts directory is removed');
});

test('real main and native tarballs install and run outside the checkout', { timeout: 600000 }, t => {
  const dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'clap-package-test-')));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  assert.ok(platforms[host], `unsupported test host ${host}`);
  const checkout = fs.existsSync(path.join(root, '..', '.github', 'scripts', 'pack-node.py'));
  const source = process.env.CLAP_CPP_PACKAGE_DIR || (checkout ? path.join(root, 'dist/clap-cpp') : root);
  assert.ok(fs.existsSync(source), `Stage packages externally first: missing ${source}`);
  const packed = pack(source, dir);
  const { files, manifest } = packed;
  checkManifest(manifest);
  for (const filename of ['index.js', 'index.d.ts', 'README.md', 'LICENSE', 'package.json',
    'test/generate-fixtures.py', 'test/tiny_models.py', 'test/types.ts', 'test/tsconfig.json']) {
    assert.ok(files.includes(filename), `missing ${filename}`);
  }
  for (const name of fs.readdirSync(path.join(root, 'test')).filter(n => n.endsWith('.test.js'))) {
    assert.ok(files.includes(`test/${name}`), `missing test/${name}`);
  }
  assert.ok(!files.some(p => /^(runtime|scripts|native|native-tests|build|src|include|cmake|ggml|thirdparty|node_modules)\//.test(p) ||
    /\.(node|so|dylib|dll|tgz)$/.test(p) || p.includes('/.git/') || p.startsWith('test/fixtures/')),
  'main package has no runtimes, build scripts, native sources or generated test artifacts');
  for (const filename of ['index.js', 'index.d.ts', 'package.json', 'README.md', ...files.filter(p => p.startsWith('test/'))]) {
    assert.deepEqual(fs.readFileSync(path.join(source, filename)), fs.readFileSync(path.join(root, filename)),
      `staged source must match checkout: ${filename}`);
  }
  const targets = process.env.CLAP_CPP_REQUIRE_ALL_RUNTIMES === '1' ? Object.keys(platforms) : [host];
  let hostArchive;
  for (const target of targets) {
    let nativeSource;
    if (target === host && process.env.CLAP_CPP_NATIVE_PACKAGE_DIR) {
      nativeSource = process.env.CLAP_CPP_NATIVE_PACKAGE_DIR;
    } else if (checkout) {
      nativeSource = path.join(root, `dist/clap-cpp-${target}`);
    } else {
      nativeSource = path.dirname(require.resolve(`@gorse/clap-cpp-${target}/package.json`, { paths: [root] }));
    }
    assert.ok(fs.existsSync(nativeSource), `missing externally staged native package: ${nativeSource}`);
    const native = pack(nativeSource, dir);
    checkNative(native, target, manifest.version);
    if (target === host) hostArchive = native.archive;
  }
  // Explicit tarballs prove local installation/inference, not registry platform
  // selection; the external test-node-install.py exercises that via a local registry.
  const installedRoot = path.join(dir, 'installed');
  run('npm', ['install', '--prefix', installedRoot, '--omit=dev', '--no-audit', '--no-fund',
    hostArchive, packed.archive]);
  const pkg = path.join(installedRoot, 'node_modules', manifest.name);
  const installedNative = path.join(installedRoot, 'node_modules', `@gorse/clap-cpp-${host}`);
  assert.equal(JSON.parse(fs.readFileSync(path.join(installedNative, 'package.json'))).version, manifest.version);
  run(process.execPath, ['-e', `
    const assert = require('node:assert/strict');
    const binding = require(${JSON.stringify(pkg)});
    assert.deepEqual(Object.keys(binding), ['CLAP']);
    assert.throws(() => new binding.CLAP('missing-model.gguf'), /.+/);
  `]);
  for (const filename of ['index.js', 'index.d.ts', 'package.json', 'test/generate-fixtures.py', 'test/tiny_models.py']) {
    assert.deepEqual(fs.readFileSync(path.join(pkg, filename)), fs.readFileSync(path.join(root, filename)));
  }
  run(python, ['test/generate-fixtures.py'], { cwd: pkg });
  run(process.execPath, ['--test', 'test/addon.test.js', 'test/surface.test.js', 'test/ffi.test.js',
    'test/runtime.test.js'], { cwd: pkg, env: { ...process.env, CLAP_CPP_MODULE: pkg } });
});
