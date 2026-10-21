"""Offline MuQ conversion metadata/config contracts (stdlib only)."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from conversion.muq import muq_inputs
revision = '0562a57814f6f8bbd9fdea0a25921a2fce1a841a'
base = {'features': ['melspec_2048'], 'n_mels': 128, 'label_rate': 25, 'hop_length': 240,
        'encoder_dim': 8, 'encoder_depth': 2, 'conv_dim': 4,
        'stat': {'melspec_2048_mean': 6.7, 'melspec_2048_std': 18.4},
        'w2v2_config': {'num_attention_heads': 2, 'intermediate_size': 16,
                       'conv_depthwise_kernel_size': 3, 'hidden_act': 'swish',
                       'position_embeddings_type': 'rotary', 'rotary_embedding_base': 10000,
                       'layer_norm_eps': 1e-5}}


class MuQConversion(unittest.TestCase):
    def convert(self, raw, ref=revision):
        with tempfile.TemporaryDirectory(prefix='MuQ metadata ') as temporary:
            source = Path(temporary)
            (source / 'config.json').write_text(json.dumps(raw))
            _, metadata = muq_inputs(source, ref)
            return metadata

    def test_metadata_records_noncommercial_license_and_revision(self):
        metadata = self.convert(base)
        self.assertEqual(metadata['general.architecture'], 'muq')
        self.assertEqual(metadata['general.license'], 'cc-by-nc-4.0')
        self.assertEqual(metadata['muq.source_revision'], revision)
        self.assertEqual(json.loads(metadata['muq.config']), base)

    def test_requires_supported_hop(self):
        raw = copy.deepcopy(base)
        raw['hop_length'] = 241
        with self.assertRaisesRegex(ValueError, 'hop_length'):
            self.convert(raw)

    def test_requires_exact_revision_token(self):
        for ref in (None, 'main', ' ' + revision, revision.upper(), revision + '0'):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.convert(base, ref)

    def test_rejects_incompatible_encoder(self):
        for field, value in [('position_embeddings_type', 'relative'), ('hidden_act', 'gelu'),
                             ('num_attention_heads', 3), ('conv_depthwise_kernel_size', 4)]:
            raw = copy.deepcopy(base)
            raw['w2v2_config'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.convert(raw)

    def test_rejects_invalid_statistics(self):
        for field, value in [('melspec_2048_mean', float('nan')), ('melspec_2048_std', 0),
                             ('melspec_2048_std', float('inf'))]:
            raw = copy.deepcopy(base)
            raw['stat'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.convert(raw)


if __name__ == '__main__':
    unittest.main()
