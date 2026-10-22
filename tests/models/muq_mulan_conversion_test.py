"""MuLan lazy discovery and local-only conversion contracts; stdlib only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

class Contracts(unittest.TestCase):
    def test_shared_muq_validation_before_schema(self):
        from conversion.muq_mulan import mulan_inputs
        from muq_conversion_test import base
        from muq_validation_cases import malformed, published
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)
            cfg=json.loads((ROOT/'tests/models/muq_mulan_config.json').read_text())
            text={'model_type':'xlm-roberta','hidden_act':'gelu','hidden_size':768,
                  'num_hidden_layers':12,'num_attention_heads':12,'intermediate_size':3072,
                  'vocab_size':4,'max_position_embeddings':514,'type_vocab_size':1,
                  'pad_token_id':1,'bos_token_id':0,'eos_token_id':2,'layer_norm_eps':1e-5}
            (source/'text-config.json').write_text(json.dumps(text))
            (source/'tokenizer.json').write_text(json.dumps({'model':{'vocab':[[w,0] for w in ['<s>','<pad>','</s>','<unk>']]}}))
            for name,raw in [('valid tiny',base),('valid published',published(base)),*malformed(base)]:
                cfg['audio_model']['model_dim']=raw['encoder_dim']
                (source/'config.json').write_text(json.dumps(cfg))
                (source/'muq-config.json').write_text(json.dumps(raw))
                with self.subTest(case=name):
                    if name.startswith('valid'):
                        mulan_inputs(source,'a'*40,None,None,None)
                    else:
                        with self.assertRaises(ValueError):
                            mulan_inputs(source,'a'*40,None,None,None)

    def test_oracle_rejects_nonfinite_components(self):
        result=subprocess.run([sys.executable,str(ROOT/'tests/models/muq_mulan_numerical_test.py')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_lazy_registration(self):
        result = subprocess.run([sys.executable, '-c', "from conversion import get_model_class; import sys; assert get_model_class('muq_mulan').__module__ == 'conversion.muq_mulan'; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'conversion.muq' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_architecture_precedence(self):
        from conversion import get_model_architecture
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'config.json').write_text((ROOT / 'tests/models/muq_mulan_config.json').read_text())
            self.assertEqual(get_model_architecture(source), 'muq_mulan')
            self.assertEqual(get_model_architecture(source, 'clamp.json'), 'muq_mulan')
            self.assertEqual(get_model_architecture(source, muq_revision='a'*40), 'muq')
            for model_type, expected in [('mert_model', 'mert'), ('mert2', 'mert2')]:
                config = json.loads((source / 'config.json').read_text()); config['model_type'] = model_type
                (source / 'config.json').write_text(json.dumps(config))
                self.assertEqual(get_model_architecture(source), expected)

    def test_requires_local_dependencies_and_revision(self):
        from conversion.muq_mulan import mulan_inputs
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'config.json').write_text((ROOT / 'tests/models/muq_mulan_config.json').read_text())
            with self.assertRaisesRegex(ValueError, 'revision'):
                mulan_inputs(source, None, None, None, None)
            with self.assertRaisesRegex(ValueError, 'muq-config'):
                mulan_inputs(source, 'a'*40, None, None, None)

    def test_unknown_tensor_rejected(self):
        from conversion.muq_mulan import tensor_name
        self.assertEqual(tensor_name('mulan.audio.model.model.conv.linear.weight'), 'conv.linear.weight')
        self.assertEqual(tensor_name('mulan.text.transformer.layers.0.0.q_scale'), 'adapter.layers.0.0.q_scale')
        self.assertIsNone(tensor_name('mulan.audio.model.model.rvq.codebook'))
        with self.assertRaisesRegex(ValueError, 'unrecognized'):
            tensor_name('mulan.audio.model.model.unexpected.weight')

    def test_full_tensor_geometry_contract(self):
        from conversion.muq_mulan import expected_shapes
        cfg = json.loads((ROOT / 'tests/models/muq_mulan_config.json').read_text())
        text = {'hidden_size':768,'vocab_size':250002,'max_position_embeddings':514,
                'type_vocab_size':1,'num_hidden_layers':12,'intermediate_size':3072}
        audio = {'encoder_dim':1024,'conv_dim':512,'encoder_depth':12,
                 'w2v2_config':{'intermediate_size':4096,'conv_depthwise_kernel_size':31}}
        shapes = expected_shapes(cfg,audio,text)
        self.assertEqual(shapes['text.embeddings.word_embeddings.weight'], (250002,768))
        self.assertEqual(shapes['adapter.layers.7.0.to_kv.weight'], (1024,768))
        self.assertEqual(shapes['adapter.layers.7.1.1.weight'], (4096,768))
        self.assertEqual(shapes['audio_proj.weight'], (768,1024))
        self.assertEqual(shapes['audio_latent.weight'], (512,768))
        self.assertEqual(shapes['conv.conv.1.conv2.weight'], (512,512,3,3))

    def test_invalid_wrapper_config(self):
        from conversion.muq_mulan import validate_config
        cfg = json.loads((ROOT / 'tests/models/muq_mulan_config.json').read_text())
        text = {'model_type': 'xlm-roberta', 'hidden_size': 768, 'num_hidden_layers': 12,
                'num_attention_heads': 12, 'intermediate_size': 3072, 'vocab_size': 250002,
                'max_position_embeddings': 514, 'type_vocab_size': 1, 'pad_token_id': 1,
                'bos_token_id': 0, 'eos_token_id': 2, 'layer_norm_eps': 1e-5, 'hidden_act': 'gelu'}
        audio = {'encoder_dim': 1024}
        validate_config(cfg, audio, text)
        for branch, field, value in [('mulan', 'clip_secs', 0), ('mulan', 'clip_secs', 0.5),
                ('mulan', 'sr', 48000), ('mulan', 'sr', 24000.0), ('audio_model', 'use_layer_idx', -1.0), ('audio_model', 'use_layer_idx', 11),
                ('audio_model', 'name', 'm-a-p/MERT-v1-330M'), ('text_model', 'name', 'Qwen'),
                ('audio_transformer', 'tf_depth', 1), ('text_transformer', 'ff_mult', float('nan'))]:
            bad = copy.deepcopy(cfg); bad[branch][field] = value
            with self.subTest(branch=branch, field=field), self.assertRaises(ValueError):
                validate_config(bad, audio, text)

if __name__ == '__main__':
    unittest.main()
