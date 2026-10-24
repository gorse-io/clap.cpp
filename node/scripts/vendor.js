'use strict';
// npm pack must be independently buildable, without Git or sibling directories.
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const pkg = path.resolve(__dirname, '..');
const root = path.resolve(pkg, '..');
const native = path.join(pkg, 'native');
const trees = ['include', 'src', 'ggml/CMakeLists.txt', 'ggml/cmake', 'ggml/include',
  'ggml/src', 'ggml/LICENSE', 'thirdparty/json/CMakeLists.txt', 'thirdparty/json/cmake',
  'thirdparty/json/include', 'thirdparty/json/nlohmann_json.natvis', 'thirdparty/json/LICENSE.MIT',
  'thirdparty/miniaudio/miniaudio.h', 'thirdparty/miniaudio/LICENSE',
  'thirdparty/icu/icu4c/source/common', 'thirdparty/icu/icu4c/source/i18n',
  'thirdparty/icu/icu4c/source/stubdata'];
const licenses = { ggml: 'ggml/LICENSE', json: 'thirdparty/json/LICENSE.MIT',
  icu: 'thirdparty/icu/LICENSE', miniaudio: 'thirdparty/miniaudio/LICENSE' };
function copy(src, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.cpSync(src, dest, { recursive: true, filter: p => !['.git', 'build', '__pycache__', 'node_modules'].includes(path.basename(p)) });
}
// Repacking an installed tarball requires no repository or Git executable.
if (!fs.existsSync(path.join(root, 'include/clap.h'))) {
  if (!fs.existsSync(path.join(native, 'include/clap.h'))) throw new Error('Native source tree missing');
  process.exit(0);
}
// Never accidentally publish source from a different submodule revision.
for (const dir of ['ggml', 'thirdparty/json', 'thirdparty/icu', 'thirdparty/miniaudio']) {
  const entry = execFileSync('git', ['ls-tree', 'HEAD', dir], { cwd: root, encoding: 'utf8' }).trim();
  const expected = entry.split(/\s+/)[2];
  const actual = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: path.join(root, dir), encoding: 'utf8' }).trim();
  if (!expected || expected !== actual) throw new Error(`${dir}: initialize the pinned submodule revision before packing`);
}
for (const p of [...trees, 'cmake/icu.cmake', 'LICENSE', ...Object.values(licenses)]) {
  if (!fs.existsSync(path.join(root, p))) throw new Error(`Missing ${p}; run git submodule update --init --recursive`);
}
fs.rmSync(native, { recursive: true, force: true });
for (const dir of trees) copy(path.join(root, dir), path.join(native, dir));
copy(path.join(root, 'cmake/icu.cmake'), path.join(native, 'cmake/icu.cmake'));
copy(path.join(root, 'thirdparty/icu/LICENSE'), path.join(native, 'thirdparty/icu/LICENSE'));
copy(path.join(root, 'LICENSE'), path.join(pkg, 'LICENSE'));
for (const [name, source] of Object.entries(licenses)) copy(path.join(root, source), path.join(pkg, 'THIRD_PARTY_LICENSES', `${name}.txt`));
copy(path.join(root, 'python/tests/tiny_models.py'), path.join(pkg, 'native-tests/tiny_models.py'));
console.log('Vendored pinned CPU sources, licenses and stdlib test fixture generator');
