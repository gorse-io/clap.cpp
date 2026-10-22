"""Malformed MuLan GGUF must fail before inference; stdlib-only default CTest."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

cli=Path(sys.argv[1]); root=Path(__file__).resolve().parents[2]
def string(s):
    b=s.encode(); return struct.pack('<Q',len(b))+b

def write(path, metadata, name='probe', dims=(1,), typ=0, value=0.):
    header=b'GGUF'+struct.pack('<IQQ',3,1,len(metadata))
    for k,v in metadata.items(): header+=string(k)+struct.pack('<I',8)+string(v)
    header+=string(name)+struct.pack('<I',len(dims))+struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',typ,0)
    data=struct.pack('<f',value) if typ==0 else b'\0'*1024
    path.write_bytes(header+b'\0'*(-len(header)%32)+data+b'\0'*1024)

with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary)/'invalid.gguf'
    cases=[('non-finite',{},0,float('nan'),'probe'),('non-finite',{},0,float('inf'),'probe'),
           ('F32/F16',{},8,0.,'probe'),('F32',{},1,0.,'frontend.window'),
           ('metadata',{},0,0.,'probe')]
    for message,meta,typ,value,name in cases:
        write(path,{'general.architecture':'muq_mulan',**meta},name=name,typ=typ,value=value,dims=(32,) if typ==8 else (1,))
        p=subprocess.run([str(cli),'--model',str(path),'--text','a'],capture_output=True,text=True)
        assert p.returncode>0 and message in p.stderr,(message,p.returncode,p.stderr)
        print(message,'rejected:',p.stderr.strip())

# Build a complete tiny GGUF entirely with stdlib. This exercises actual loader
# geometry/config failures without Torch or downloaded source/model fixtures.
sys.path.insert(0,str(root))
from conversion.muq_mulan import expected_shapes
cfg=json.loads((root/'tests/models/muq_mulan_config.json').read_text())
cfg['mulan'].update(clip_secs=1,dim_latent=6);cfg['audio_model']['model_dim']=8
for branch,depth in [('audio_transformer',0),('text_transformer',1)]:
    cfg[branch].update(dim=8,tf_depth=depth,heads=2,dim_head=4,ff_mult=4)
audio={'features':['melspec_2048'],'n_mels':128,'hop_length':240,'label_rate':25,
       'encoder_dim':8,'encoder_depth':1,'conv_dim':2,'stat':{'melspec_2048_mean':0,'melspec_2048_std':1},
       'w2v2_config':{'num_attention_heads':2,'intermediate_size':16,'conv_depthwise_kernel_size':3,
       'hidden_act':'swish','position_embeddings_type':'rotary','rotary_embedding_base':10000,'layer_norm_eps':1e-5}}
text={'model_type':'xlm-roberta','hidden_size':8,'num_hidden_layers':1,'num_attention_heads':2,
      'intermediate_size':16,'vocab_size':6,'max_position_embeddings':514,'type_vocab_size':1,
      'pad_token_id':1,'bos_token_id':0,'eos_token_id':2,'hidden_act':'gelu','layer_norm_eps':1e-5}
vocab=['<s>','<pad>','</s>','<unk>','▁','a']
tokenizer={'model':{'unk_id':3,'vocab':[[w,0.] for w in vocab]},
 'pre_tokenizer':{'type':'Sequence','pretokenizers':[{'type':'WhitespaceSplit'},{'type':'Metaspace','replacement':'▁','add_prefix_space':True}]},
 'post_processor':{'type':'TemplateProcessing','single':[{'SpecialToken':{'id':'<s>','type_id':0}},{'Sequence':{'id':'A','type_id':0}},{'SpecialToken':{'id':'</s>','type_id':0}}],
 'special_tokens':{'<s>':{'ids':[0]},'</s>':{'ids':[2]}}}}
base_meta={'general.architecture':'muq_mulan','muq.config':json.dumps(audio),'muq_mulan.config':json.dumps(cfg),
           'muq_mulan.text_config':json.dumps(text),'muq_mulan.tokenizer':json.dumps(tokenizer)}
shapes=expected_shapes(cfg,audio,text)
def complete(path,meta,geometry,alter=None,half_name=None):
    import math
    header=b'GGUF'+struct.pack('<IQQ',3,len(geometry),len(meta));payload=b''
    for k,v in meta.items():header+=string(k)+struct.pack('<I',8)+string(v)
    for name,shape in geometry.items():
        header+=string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*reversed(shape))+struct.pack('<IQ',int(name==half_name),len(payload))
        elements=math.prod(shape)
        value=1. if name.endswith(('running_var','learned_gamma')) else 0.
        if alter and name==alter[0]:value=alter[1]
        payload+=struct.pack('<e' if name==half_name else '<f',value)*elements
        payload+=b'\0'*(-len(payload)%32)
    path.write_bytes(header+b'\0'*(-len(header)%32)+payload)
with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary)/'complete.gguf'
    complete(path,base_meta,shapes)
    baseline=subprocess.run([str(cli),'--model',str(path),'--text',''],capture_output=True,text=True)
    assert baseline.returncode==0,(baseline.returncode,baseline.stderr)
    assert json.loads(baseline.stdout)==[0.]*6
    cases=[]
    for branch,key,value in [('audio_model','use_layer_idx',11),('audio_transformer','tf_depth',1),('mulan','clip_secs',0),('mulan','clip_secs',0.5),('mulan','sr',24000.0),('audio_model','use_layer_idx',-1.0)]:
        bad=copy.deepcopy(cfg);bad[branch][key]=value
        cases.append((dict(base_meta,**{'muq_mulan.config':json.dumps(bad)}),shapes,None,'MuLan'))
    for key,value in [('is_decoder',True),('layer_norm_eps',-1),('num_attention_heads',3),('pad_token_id',-1),('pad_token_id',1.0)]:
        bad=dict(text,**{key:value})
        cases.append((dict(base_meta,**{'muq_mulan.text_config':json.dumps(bad)}),shapes,None,'MuLan'))
    bad=dict(shapes);bad['text_proj.weight']=(7,8);cases.append((base_meta,bad,None,'shape'))
    bad=dict(shapes);del bad['audio_proj.bias'];cases.append((base_meta,bad,None,'missing tensor'))
    bad=dict(shapes);bad['unexpected.weight']=(1,);cases.append((base_meta,bad,None,'unrecognized'))
    cases.append((base_meta,shapes,('adapter.layers.0.0.q_scale',float('nan')),'non-finite'))
    cases.append((base_meta,shapes,('conv.conv.0.bn1.running_var',-1.),'variance'))
    for meta,geometry,alter,message in cases:
        complete(path,meta,geometry,alter)
        proc=subprocess.run([str(cli),'--model',str(path),'--text','a'],capture_output=True,text=True)
        assert proc.returncode>0 and message in proc.stderr,(message,proc.returncode,proc.stderr)
    overflow=copy.deepcopy(cfg);overflow['text_transformer']['tf_depth']=4294967296
    complete(path,dict(base_meta,**{'muq_mulan.config':json.dumps(overflow)}),
             {k:v for k,v in shapes.items() if not k.startswith('adapter.')})
    proc=subprocess.run([str(cli),'--model',str(path),'--text','a'],capture_output=True,text=True)
    assert proc.returncode>0 and 'dimensions' in proc.stderr, ('depth overflow must fail',proc.returncode,proc.stderr)
    complete(path,base_meta,shapes,half_name='conformer.layer_norm.weight')
    proc=subprocess.run([str(cli),'--model',str(path),'--text','a'],capture_output=True,text=True)
    assert proc.returncode>0 and 'F32' in proc.stderr, ('F16 norm must fail',proc.returncode,proc.stderr)
    print('Complete tiny MuLan: legacy tokenizer accepted;',len(cases),'malformed contracts rejected')

# Shared dependency validation must run before tensor shape lookup/narrowing for
# BOTH architectures. Valid published dimensions reach shape checking, while
# malformed metadata must fail specifically in the shared MuQ validator.
from muq_validation_cases import malformed, published
failures=[]
architectures = [sys.argv[2]] if len(sys.argv)>2 else ['muq_mulan','muq']
with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary)/'dependency.gguf'
    for arch in architectures:
        for name,raw in [('valid tiny',audio),('valid published',published(audio)),*malformed(audio)]:
            meta=dict(base_meta, **{'general.architecture':arch,'muq.config':json.dumps(raw)})
            complete(path,meta,shapes)
            proc=subprocess.run([str(cli),'--model',str(path),'--text','a'],capture_output=True,text=True,timeout=15)
            if name=='valid tiny':
                ok=proc.returncode==0 if arch=='muq_mulan' else 'no text encoder' in proc.stderr
            elif name=='valid published':
                ok=proc.returncode>0 and ('shape' in proc.stderr or 'missing tensor' in proc.stderr)
            else:
                ok=proc.returncode>0 and ('invalid MuQ' in proc.stderr or '[json.exception' in proc.stderr)
            if not ok: failures.append((arch,name,proc.returncode,proc.stderr.strip()))
            print(arch,name,'PASS' if ok else 'FAIL',proc.stderr.strip(),flush=True)
assert not failures, failures
