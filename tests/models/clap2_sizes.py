#!/usr/bin/env python3
"""Offline published-size configuration and source-identity regressions."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conversion.clap2 import clap2_inputs, expected_tensor_shapes, validate_config
from clap2_fixture import model_data


class PublishedSizes(unittest.TestCase):
    def test_published_configs_and_identity(self):
        fixtures = json.loads(Path(__file__).with_suffix('.json').read_text())
        for size, model in fixtures['models'].items():
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temp:
                source = Path(temp)
                config = model['config']
                validate_config(config, fixtures['processor'])
                shapes = expected_tensor_shapes(config)
                self.assertEqual(len(shapes), model['tensor_count'])
                schema = {name: list(shape) for name, shape in shapes.items()}
                digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                # Digest of every tensor name/shape from the pinned published
                # safetensors header, independent of the converter's schema.
                self.assertEqual(digest, model['tensor_shapes_sha256'])
                self.assertEqual(shapes['text_projection.weight'],
                                 (config['projection_dim'], config['text_config']['hidden_size']))
                # Use a complete synthetic byte-BPE vocabulary to keep this
                # discovery test stdlib-only; published tokenizer parity is separate.
                tokenizer = model_data()[2]
                config = copy.deepcopy(config)
                for key in ('vocab_size', 'bos_token_id', 'eos_token_id', 'pad_token_id'):
                    config['text_config'][key] = model_data()[0]['text_config'][key]
                for name, data in (('config.json', config), ('processor_config.json', fixtures['processor']),
                                   ('tokenizer.json', tokenizer)):
                    (source / name).write_text(json.dumps(data))
                (source / 'model.safetensors').touch()
                _, metadata, _ = clap2_inputs(source)
                self.assertEqual(metadata['general.name'], model['identity']['repo'])
                self.assertEqual(metadata['clap2.source_repo'], model['identity']['repo'])
                self.assertEqual(metadata['clap2.source_revision'], model['identity']['revision'])
                override = 'a' * 40
                self.assertEqual(clap2_inputs(source, override)[1]['clap2.source_revision'], override)

    def test_custom_model_does_not_claim_large_checkpoint(self):
        config, processor, tokenizer, _ = model_data()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            for name, data in (('config.json', config), ('processor_config.json', processor),
                               ('tokenizer.json', tokenizer)):
                (source / name).write_text(json.dumps(data))
            (source / 'model.safetensors').touch()
            _, metadata, _ = clap2_inputs(source)
            self.assertEqual(metadata['general.name'], 'CLAP2')
            self.assertNotIn('clap2.source_repo', metadata)
            self.assertNotIn('clap2.source_revision', metadata)
            self.assertEqual(clap2_inputs(source, 'b' * 40)[1]['clap2.source_revision'], 'b' * 40)


if __name__ == '__main__':
    unittest.main()
