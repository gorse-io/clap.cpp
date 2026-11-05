'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

for (const platform of ['darwin', 'linux', 'win32']) {
  test(`build configures platform defaults for ${platform}`, t => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clap-build-platform-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    fs.mkdirSync(path.join(dir, 'scripts'));
    fs.mkdirSync(path.join(dir, 'native'));
    fs.copyFileSync(path.join(__dirname, '../scripts/build.js'), path.join(dir, 'scripts/build.js'));
    fs.writeFileSync(path.join(dir, 'native/CMakeLists.txt'), '# test fixture');
    const preload = path.join(dir, 'capture-cmake.cjs');
    fs.writeFileSync(preload, `
      Object.defineProperty(process, 'platform', { value: ${JSON.stringify(platform)} });
      require('node:child_process').spawnSync = (command, args) => {
        require('node:fs').writeFileSync('configure.json', JSON.stringify({ command, args }));
        return { status: 37 };
      };
    `);
    const result = spawnSync(process.execPath, ['--require', preload, 'scripts/build.js'], {
      cwd: dir, encoding: 'utf8',
    });
    // Stop at the configure boundary: no compiler or target-platform host needed.
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /CMake failed \(37\)/);
    const { command, args } = JSON.parse(fs.readFileSync(path.join(dir, 'configure.json'), 'utf8'));
    assert.equal(command, 'cmake');
    assert.equal(args[0], '-S');
    const expectedMetal = platform === 'darwin' ? 'ON' : 'OFF';
    for (const name of ['GGML_METAL', 'GGML_METAL_EMBED_LIBRARY']) {
      assert.deepEqual(args.filter(arg => arg.startsWith(`-D${name}=`)), [`-D${name}=${expectedMetal}`]);
    }
    for (const name of ['GGML_NATIVE', 'GGML_BACKEND_DL', 'GGML_OPENMP', 'GGML_BUILD_TESTS',
      'GGML_BUILD_EXAMPLES', 'GGML_CUDA', 'GGML_HIP', 'GGML_VULKAN', 'GGML_SYCL', 'GGML_OPENCL',
      'GGML_OPENVINO', 'GGML_RPC', 'GGML_MUSA', 'GGML_WEBGPU', 'GGML_ZDNN', 'GGML_VIRTGPU',
      'GGML_VIRTGPU_BACKEND', 'GGML_ET', 'GGML_HEXAGON', 'GGML_ZENDNN', 'GGML_BLAS',
      'GGML_ACCELERATE', 'GGML_SSE42', 'GGML_AVX', 'GGML_AVX2', 'GGML_AVX_VNNI', 'GGML_BMI2',
      'GGML_FMA', 'GGML_F16C', 'GGML_AVX512', 'GGML_AVX512_VBMI', 'GGML_AVX512_VNNI',
      'GGML_AVX512_BF16', 'GGML_AMX_TILE', 'GGML_AMX_INT8', 'GGML_AMX_BF16']) {
      assert.ok(args.includes(`-D${name}=OFF`), name);
    }
    assert.ok(args.includes('-DGGML_CPU=ON'));
    assert.ok(args.includes('-DBUILD_SHARED_LIBS=ON'));
    assert.ok(args.includes('-DCLAP_STATIC_DEPENDENCIES=ON'));
  });
}
