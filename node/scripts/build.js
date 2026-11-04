'use strict';
// Build the public C ABI with the repository's one native build graph.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { spawnSync } = require('node:child_process');
const pkg = path.resolve(__dirname, '..');
const checkout = path.resolve(pkg, '..');
const source = fs.existsSync(path.join(checkout, 'include/clap.h')) &&
  fs.existsSync(path.join(checkout, 'CMakeLists.txt')) ? checkout : path.join(pkg, 'native');
const output = path.join(pkg, 'build');
const build = path.join(output, 'cmake');
const runtime = path.join(output, 'runtime');
function run(args) {
  const result = spawnSync('cmake', args, { cwd: pkg, stdio: 'inherit' });
  if (result.error) throw new Error(`Cannot run CMake: ${result.error.message}. Install CMake >=3.20 and a C++20 compiler.`);
  if (result.status !== 0) throw new Error(`CMake failed (${result.signal || result.status}); native library was not installed successfully.`);
}
function main() {
  if (process.argv.slice(2).some(arg => arg !== '--rebuild')) throw new Error('Usage: node scripts/build.js [--rebuild]');
  // Never follow caller-created output symlinks, including during rebuild.
  for (const dir of [output, build, runtime]) {
    let stat;
    try { stat = fs.lstatSync(dir); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (stat && (stat.isSymbolicLink() || !stat.isDirectory())) throw new Error(`Refusing unsafe output: ${dir}`);
  }
  if (!fs.existsSync(path.join(source, 'CMakeLists.txt'))) throw new Error('Native source snapshot is missing; use an npm source tarball or initialize the checkout submodules.');
  if (process.argv.includes('--rebuild')) {
    for (const dir of [build, runtime]) fs.rmSync(dir, { recursive: true, force: true });
  }
  fs.mkdirSync(output, { recursive: true });
  const off = ['GGML_NATIVE', 'GGML_BACKEND_DL', 'GGML_OPENMP', 'GGML_BUILD_TESTS', 'GGML_BUILD_EXAMPLES',
    'GGML_METAL', 'GGML_CUDA', 'GGML_HIP', 'GGML_VULKAN', 'GGML_SYCL', 'GGML_OPENCL', 'GGML_OPENVINO',
    'GGML_RPC', 'GGML_MUSA', 'GGML_WEBGPU', 'GGML_ZDNN', 'GGML_VIRTGPU', 'GGML_VIRTGPU_BACKEND',
    'GGML_ET', 'GGML_HEXAGON', 'GGML_ZENDNN', 'GGML_BLAS', 'GGML_ACCELERATE',
    ...['SSE42', 'AVX', 'AVX2', 'AVX_VNNI', 'BMI2', 'FMA', 'F16C', 'AVX512', 'AVX512_VBMI',
      'AVX512_VNNI', 'AVX512_BF16', 'AMX_TILE', 'AMX_INT8', 'AMX_BF16'].map(isa => `GGML_${isa}`)];
  run(['-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
    '-DCLAP_STATIC_DEPENDENCIES=ON', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DCLAP_BUILD_CLI=OFF',
    '-DBUILD_TESTING=OFF', '-DCLAP_USE_SYSTEM_ICU=OFF', '-DGGML_CPU=ON',
    `-DCMAKE_INSTALL_PREFIX=${runtime}`, '-DCMAKE_INSTALL_LIBDIR=.', '-DCMAKE_INSTALL_BINDIR=.',
    ...off.map(name => `-D${name}=OFF`)]);
  const jobs = process.env.CMAKE_BUILD_PARALLEL_LEVEL || String(Math.min(os.availableParallelism(), 4));
  if (!/^[1-9]\d*$/.test(jobs)) throw new Error('CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer');
  run(['--build', build, '--config', 'Release', '--target', 'clap', '--parallel', jobs]);
  run(['--install', build, '--config', 'Release', '--component', 'runtime']);
  fs.copyFileSync(path.join(source, 'LICENSE'), path.join(runtime, 'licenses', 'clap.txt'));
  const library = process.platform === 'win32' ? 'clap.dll' : process.platform === 'darwin' ? 'libclap.dylib' : 'libclap.so';
  if (!fs.existsSync(path.join(runtime, library))) throw new Error(`CMake did not install ${library}`);
}
try { main(); } catch (error) { console.error(`clap-cpp: ${error.message}`); process.exitCode = 1; }
