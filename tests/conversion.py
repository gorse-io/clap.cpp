"""Test lazy converter discovery without requiring model dependencies."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ConversionImportsTest(unittest.TestCase):
    def run_python(self, code):
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_entry_point_does_not_import_model_modules(self):
        self.run_python('''
import sys
import convert_hf_to_gguf
assert not any('conversion.' + name in sys.modules for name in ('clap', 'clamp3', 'mert2'))
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
''')

    def test_loads_only_selected_converter(self):
        for name in ('clap', 'clamp3', 'mert2'):
            with self.subTest(model=name):
                self.run_python(f'''
import sys
from conversion import get_model_class
converter = get_model_class({name!r})
from conversion import ModelBase
assert issubclass(converter, ModelBase)
assert converter.__module__ == 'conversion.' + {name!r}
assert converter is get_model_class({name!r})
for name in ('clap', 'clamp3', 'mert2'):
    assert ('conversion.' + name in sys.modules) == (name == {name!r})
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
''')

    def test_architecture_selection_preserves_checkpoint_rules(self):
        self.run_python('''
import json
from pathlib import Path
import tempfile
from conversion import get_model_architecture
with tempfile.TemporaryDirectory() as directory:
    source = Path(directory)
    assert get_model_architecture(source) == 'clap'
    assert get_model_architecture(source, 'combined.json') == 'clamp3'
    checkpoint = source / 'weights.pth'
    checkpoint.touch()
    assert get_model_architecture(checkpoint) == 'clamp3'
    (source / 'config.json').write_text(json.dumps({'model_type': 'mert2'}))
    assert get_model_architecture(source) == 'mert2'
    assert get_model_architecture(source, 'combined.json') == 'mert2'
''')

    def test_rejects_unknown_model(self):
        self.run_python('''
from conversion import get_model_class
try:
    get_model_class('unknown')
except NotImplementedError as error:
    assert 'unknown' in str(error)
else:
    raise AssertionError('unknown architecture accepted')
''')


if __name__ == '__main__':
    unittest.main()
