"""Tiny MuLan storage validation, requires Torch/torchaudio; no model downloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from conversion.muq_mulan import MuQMuLanModel, expected_shapes
from muq_conversion_test import base as audio
cfg=json.loads((ROOT/'tests/models/muq_mulan_config.json').read_text())
cfg['mulan'].update(clip_secs=1,dim_latent=6);cfg['audio_model']['model_dim']=8
for branch,depth in [('audio_transformer',0),('text_transformer',1)]:
    cfg[branch].update(dim=8,tf_depth=depth,heads=2,dim_head=4,ff_mult=4)
text={'hidden_size':8,'num_hidden_layers':1,'intermediate_size':16,'vocab_size':6,
      'max_position_embeddings':514,'type_vocab_size':1}
shapes=expected_shapes(cfg,audio,text)
def original(name):
    for prefix,target in [('text.embeddings.','mulan.text.model.embeddings.'),('text.encoder.','mulan.text.model.encoder.'),
            ('adapter.','mulan.text.transformer.'),('audio_proj.','mulan.audio.proj.'),
            ('text_proj.','mulan.text_to_latents.'),('audio_latent.','mulan.audio_to_latents.')]:
        if name.startswith(prefix): return target+name[len(prefix):]
    return 'mulan.audio.model.model.'+name

class Storage(unittest.TestCase):
    def test_half_overflow_before_destination_open(self):
        weights={original(n):torch.ones(s) for n,s in shapes.items() if not n.startswith('frontend.')}
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);checkpoint=tmp/'weights';checkpoint.touch()
            metadata={'general.architecture':'muq_mulan','muq.config':json.dumps(audio),
                'muq_mulan.config':json.dumps(cfg),'muq_mulan.text_config':json.dumps(text)}
            dst=tmp/'result.gguf';dst.write_bytes(b'prior destination')
            matrix=weights['mulan.audio.proj.weight'];matrix[0,0]=100000
            with patch('conversion.muq_mulan.mulan_inputs',return_value=(checkpoint,metadata)),patch('conversion.muq_mulan.load_checkpoint',return_value=weights):
                with patch('conversion.base.tempfile.mkstemp',side_effect=AssertionError('destination opened before validation')):
                    with self.assertRaisesRegex(ValueError,'F16'):
                        MuQMuLanModel(tmp,dst,'f16').write()
                self.assertEqual(dst.read_bytes(),b'prior destination')
                MuQMuLanModel(tmp,dst,'f32').write()  # Same finite matrix is valid F32.
                matrix[0,0]=1
                weights['mulan.audio.proj.bias'][0]=100000  # 1D remains F32.
                MuQMuLanModel(tmp,dst,'f16').write()
                self.assertTrue(dst.read_bytes().startswith(b'GGUF'))

if __name__=='__main__':unittest.main()
