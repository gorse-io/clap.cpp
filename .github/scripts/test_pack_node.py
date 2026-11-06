"""Unit-test staging metadata; the payload bytes here are not executable binaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('pack_node', Path(__file__).with_name('pack-node.py'))
assert SPEC is not None and SPEC.loader is not None
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


class NodePackageTest(unittest.TestCase):
    def test_platform_packages_and_exact_optional_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = root / 'node'
            (node / 'test').mkdir(parents=True)
            (root / 'python/tests').mkdir(parents=True)
            (node / 'package.json').write_text(json.dumps({
                'name': 'clap-cpp', 'version': '1.2.3-rc.1', 'license': 'Apache-2.0',
                'repository': {'url': 'https://github.com/gorse-io/clap.cpp'},
                'files': ['index.js', 'index.d.ts', 'test'],
            }))
            for filename in ['index.js', 'index.d.ts', 'README.md']:
                (node / filename).write_text('unit fixture')
            (node / 'test/probe.test.js').write_text('unit fixture')
            (node / 'test/probe.log').write_text('must not ship')
            (root / 'LICENSE').write_text('unit license')
            (root / 'python/tests/tiny_models.py').write_text('unit fixture generator')
            runtimes = root / 'runtimes'
            for target, library in PACK.TARGETS.items():
                runtime = runtimes / target
                (runtime / 'licenses').mkdir(parents=True)
                (runtime / library).write_bytes(b'non-executable unit fixture')
                for name in ['ggml', 'icu', 'json', 'miniaudio']:
                    (runtime / f'licenses/{name}.txt').write_text('unit license')
            output = root / 'packages'
            staged = PACK.stage(runtimes, output, list(PACK.TARGETS), root)
            self.assertEqual(len(staged), 5)
            main = json.loads((output / 'clap-cpp/package.json').read_text())
            self.assertEqual(main['optionalDependencies'], {
                f'@gorse/clap-cpp-{target}': '1.2.3-rc.1' for target in PACK.TARGETS
            })
            self.assertFalse((output / 'clap-cpp/runtime').exists())
            self.assertTrue((output / 'clap-cpp/test/tiny_models.py').is_file())
            self.assertFalse((output / 'clap-cpp/test/probe.log').exists())
            for target, library in PACK.TARGETS.items():
                package = output / f'clap-cpp-{target}'
                metadata = json.loads((package / 'package.json').read_text())
                os_name, arch = target.split('-')
                self.assertEqual(metadata['os'], [os_name])
                self.assertEqual(metadata['cpu'], [arch])
                self.assertEqual(metadata['version'], main['version'])
                self.assertEqual(metadata.get('libc'), ['glibc'] if os_name == 'linux' else None)
                self.assertTrue((package / f'runtime/{library}').is_file())
                self.assertFalse((package / 'index.js').exists())
            # An incomplete input must fail before replacing reviewed output.
            (runtimes / 'linux-x64/libclap.so').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing linux-x64 runtime input'):
                PACK.stage(runtimes, output, ['linux-x64'], root)
            self.assertTrue((output / 'clap-cpp/package.json').is_file())


if __name__ == '__main__':
    unittest.main()
