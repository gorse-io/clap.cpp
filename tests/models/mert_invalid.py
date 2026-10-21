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
       ('num_feat_extract_layers',6,'convolution geometry')]
with tempfile.TemporaryDirectory(prefix='mert-invalid-') as temp:
    target=Path(temp)/'invalid.gguf'
    for key,value,message in cases:
        config=copy.deepcopy(base);config[key]=value;model(target,config)
        result=subprocess.run([cli,'-m',str(target),'-p','invalid'],capture_output=True,text=True)
        assert result.returncode>0 and message in result.stderr, (key,result.returncode,result.stderr)
    print(f'{len(cases)} invalid MERT metadata cases rejected')
