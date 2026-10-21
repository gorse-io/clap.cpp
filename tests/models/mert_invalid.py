#!/usr/bin/env python3
"""Always-on metadata rejection tests; no torch, models or network required."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

cli=sys.argv[1]
base=dict(model_type='mert_model',sample_rate=24000,feature_extractor_cqt=False,
          deepnorm=False,do_stable_layer_norm=False,feat_extract_norm='group',feat_proj_layer_norm=True,
          hidden_act='gelu',feat_extract_activation='gelu',hidden_size=16,intermediate_size=32,
          num_hidden_layers=2,num_attention_heads=2,num_conv_pos_embeddings=8,
          num_conv_pos_embedding_groups=2,conv_dim=[8]*7,conv_kernel=[10,3,3,3,3,2,2],
          conv_stride=[5,2,2,2,2,2,2],layer_norm_eps=1e-5,conv_bias=False)
def string(x):
    x=x.encode();return struct.pack('<Q',len(x))+x

def model(path,config):
    metadata={'general.architecture':'mert','mert.config':json.dumps(config)}
    data=b'GGUF'+struct.pack('<IQQ',3,0,len(metadata))
    for k,v in metadata.items():data+=string(k)+struct.pack('<I',8)+string(v)
    path.write_bytes(data+b'\0'*(-len(data)%32))

cases=[('sample_rate',48000,'24 kHz'),('hidden_size',0,'invalid MERT'),
       ('num_attention_heads',3,'groups'),('num_conv_pos_embedding_groups',3,'groups'),
       ('num_conv_pos_embeddings',0,'invalid MERT'),('intermediate_size',2.5,'invalid MERT'),
       ('num_hidden_layers',True,'invalid MERT'),('conv_dim',[],'geometry'),
       ('conv_kernel',[-1]*7,'dimension'),('conv_stride',[0]*7,'dimension'),
       ('conv_dim',[8.5]*7,'dimension'),('conv_stride',[100000]*7,'receptive'),
       ('layer_norm_eps',0,'layer_norm_eps'),('feature_extractor_cqt',True,'variant'),
       ('deepnorm',True,'variant'),('do_stable_layer_norm',True,'variant'),
       ('feat_extract_norm','layer','variant'),('hidden_act','relu','variant'),
       ('conv_pos_batch_norm',True,'variant'),('feat_proj_layer_norm',False,'variant'),
       ('num_feat_extract_layers',6,'convolution geometry'),
       ('num_feat_extract_layers',7.5,'convolution geometry'),
       ('hidden_size',2**32+16,'invalid MERT'),
       ('conv_dim',[2**32+8]*7,'dimension'),
       ('conv_bias',0,'invalid MERT boolean'),
       ('feat_proj_layer_norm',1,'invalid MERT boolean'),
       ('feature_extractor_cqt',0,'invalid MERT boolean'),
       ('layer_norm_eps',True,'layer_norm_eps'),
       ('layer_norm_eps',1e100,'layer_norm_eps'),
       ('layer_norm_eps',1e-100,'layer_norm_eps')]
with tempfile.TemporaryDirectory(prefix='mert-invalid-') as temp:
    target=Path(temp)/'invalid.gguf'
    for key,value,message in cases:
        config=copy.deepcopy(base);config[key]=value;model(target,config)
        result=subprocess.run([cli,'-m',str(target),'-p','invalid'],capture_output=True,text=True)
        assert result.returncode>0 and message in result.stderr, (key,result.returncode,result.stderr)
    print(f'{len(cases)} invalid MERT metadata cases rejected')

# Converter validation must also run without torch and before checkpoint loading.
import importlib.util
spec = importlib.util.spec_from_file_location('converter', Path(__file__).resolve().parents[2]/'convert_hf_to_gguf.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
preprocessor = dict(feature_extractor_type='Wav2Vec2FeatureExtractor', feature_size=1,
                    sampling_rate=24000, do_normalize=True, padding_value=0,
                    padding_side='right', return_attention_mask=True)
revision = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
count = 0
with tempfile.TemporaryDirectory(prefix='mert-converter-invalid-') as temp:
    source = Path(temp)
    def reject(config, processor, sha):
        global count
        (source/'config.json').write_text(json.dumps(config))
        (source/'preprocessor_config.json').write_text(json.dumps(processor))
        try:
            converter.mert_inputs(source, sha, object())
        except ValueError:
            count += 1
        else:
            raise AssertionError('converter accepted invalid metadata')
    for key, value, _ in cases:
        config = copy.deepcopy(base); config[key] = value
        reject(config, preprocessor, revision)
    for key, value in [('sampling_rate',48000), ('do_normalize',False),
                        ('feature_size',2), ('return_attention_mask',False),
                        ('feature_size',True), ('return_attention_mask',1),
                        ('sampling_rate',24000.0), ('do_normalize',1), ('padding_value',False)]:
        processor = dict(preprocessor); processor[key] = value
        reject(base, processor, revision)
    for sha in [None, '', 'a'*39, 'x'*40]:
        reject(base, preprocessor, sha)
print(f'{count} invalid MERT converter cases rejected before checkpoint loading')

# Only the checkpoint-loading boundary is stubbed: test the real converter's
# positional-weight representation validation without making Torch a default dependency.
class Checkpoint:
    def __init__(self, names):
        self.weights = {'encoder.pos_conv_embed.conv.'+name: object() for name in names}
    def load(self, *args, **kwargs):
        assert kwargs['weights_only'] and kwargs['mmap']
        return dict(self.weights)
    def _weight_norm(self, *args):
        raise AssertionError('invalid weight-norm pair reached materialization')

representations = [
    ['weight_g'], ['weight_v'], ['parametrizations.weight.original0'],
    ['parametrizations.weight.original1'], ['weight','weight_g','weight_v'],
    ['weight_g','weight_v','parametrizations.weight.original0','parametrizations.weight.original1'],
    [],
]
with tempfile.TemporaryDirectory(prefix='mert-converter-weights-') as temp:
    source = Path(temp)
    (source/'config.json').write_text(json.dumps(base))
    (source/'preprocessor_config.json').write_text(json.dumps(preprocessor))
    for names in representations:
        try:
            converter.mert_inputs(source, revision, Checkpoint(names))
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid positional weight representation: {names}')
print(f'{len(representations)} invalid MERT positional weight representations rejected')
