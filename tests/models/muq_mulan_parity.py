"""Tiny official-component joint oracle; synthetic, not published quality evidence."""
import ast
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import torch
from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors, AddedToken
from tokenizers.models import Unigram
from transformers import XLMRobertaConfig
from muq_mulan_reference import Joint

build, out, official, muq_source = map(Path, sys.argv[1:])
root = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(root/'tests/models/muq_mulan_storage_test.py')],check=True)
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
torch.set_num_threads(2); torch.manual_seed(24680)
cfg = json.loads((root / 'tests/models/muq_mulan_config.json').read_text())
cfg['mulan'].update(clip_secs=1, dim_latent=6)
cfg['audio_model']['model_dim'] = 8
for branch, depth in [('audio_transformer',0), ('text_transformer',2)]:
    cfg[branch].update(dim=8, tf_depth=depth, heads=2, dim_head=4, ff_mult=4)
audio_cfg = {'encoder_dim':8,'encoder_depth':2,'conv_dim':2,'n_mels':128,'hop_length':240,
 'label_rate':25,'features':['melspec_2048'],'is_flash':False,
 'stat':{'melspec_2048_mean':6.768444971712967,'melspec_2048_std':18.417922652295623},
 'w2v2_config':{'hidden_size':8,'num_hidden_layers':2,'num_attention_heads':2,'intermediate_size':16,
 'hidden_act':'swish','position_embeddings_type':'rotary','conv_depthwise_kernel_size':3,
 'rotary_embedding_base':10000,'num_conv_pos_embeddings':8,'num_conv_pos_embedding_groups':2,
 'layer_norm_eps':1e-5,'hidden_dropout':0.,'attention_dropout':0.,'activation_dropout':0.,'conformer_conv_dropout':0.,'layerdrop':0.}}
vocab = [('<s>',0.),('<pad>',0.),('</s>',0.),('<unk>',0.),('<mask>',0.),('▁',-3.),('a',-2.),('b',-2.1),('c',-2.2),('▁a',-.2),('▁ab',-.1),('é',-.4),('▁é',-.1),('音',-.3),('乐',-.3),('▁音',-.1),('中',-.3),('文',-.2)]
# Reuse the existing deterministic normalization fixture, without importing the
# executable CLaMP3 oracle (which would construct/run unrelated models).
_tree=ast.parse((root/'tests/models/clamp3_parity.py').read_text())
_fn=next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='tiny_charsmap')
_scope={'struct':struct}; exec(compile(ast.Module(body=[_fn],type_ignores=[]),'tiny_charsmap','exec'),_scope)
tok = Tokenizer(Unigram(vocab, unk_id=3)); tok.normalizer=normalizers.Precompiled(_scope['tiny_charsmap']())
tok.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(),pre_tokenizers.Metaspace(replacement='▁',prepend_scheme='always')])
tok.add_special_tokens([AddedToken(x, special=True, lstrip=False) for x in ('<s>','<pad>','</s>','<unk>','<mask>')])
tok.post_processor=processors.TemplateProcessing(single='<s> $A </s>', special_tokens=[('<s>',0),('</s>',2)])
tok.save(str(out/'tokenizer.json'))
text_cfg = XLMRobertaConfig(vocab_size=len(vocab),hidden_size=8,num_hidden_layers=2,num_attention_heads=2,
 intermediate_size=16,max_position_embeddings=514,type_vocab_size=1,pad_token_id=1,bos_token_id=0,eos_token_id=2,
 hidden_act='gelu',layer_norm_eps=1e-5,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.).to_dict()
for file, value in [('config.json',cfg),('muq-config.json',audio_cfg),('text-config.json',text_cfg)]:
    (out/file).write_text(json.dumps(value))
model = Joint(cfg,audio_cfg,text_cfg,official,muq_source).eval()
with torch.no_grad():
    for name,p in model.named_parameters():
        if 'learned_gamma' in name: p.uniform_(.65,1.35)
        if name.endswith(('q_scale','k_scale')): p.uniform_(.4,1.7)
    for module in model.modules():
        if isinstance(module,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d)):
            module.running_mean.uniform_(-.3,.3); module.running_var.uniform_(.4,1.6)
            module.weight.uniform_(.7,1.3); module.bias.uniform_(-.2,.2)
# Match official wrapper checkpoint nesting; no custom source constructors executed.
weights = {}
for name,value in model.state_dict().items():
    if name.startswith('audio.model.'): name='audio.model.model.'+name[len('audio.model.'):]
    weights['mulan.'+name]=value
# Known unused tensors exercise intentional filtering.
weights['mulan.audio.aggregator.weight']=torch.zeros(1,25,1)
weights['mulan.audio.aggregator.bias']=torch.zeros(1)
weights['mulan.contrast.temperature']=torch.tensor(1.)
torch.save(weights,out/'pytorch_model.bin')
texts=['classical piano a b','<pad>ab','ab c','e\u0301 音乐','中文音乐','', 'a <mask> b','🫨', '<s>ab</s>']
def wave(n):
    t=torch.arange(n)/24000
    x=.17*torch.sin(2*torch.pi*(180*t+180*t*t))+.08*torch.cos(2*torch.pi*733*t)
    x*=torch.linspace(.1,2.7,n); x[-1]+=.31
    return x
volume=wave(6001)+.002*torch.randn(6001,generator=torch.Generator().manual_seed(20261022))
waves=[('quiet amplitude',volume*.3),('loud amplitude',volume*3),('513 one wrap',wave(513)),('below half',wave(6001)),('half',wave(12000)),('exact',wave(24000)),('two exact',wave(48000)),('multi tail',wave(55031))]
for dtype in ('f32','f16'):
    current=copy.deepcopy(model)
    if dtype=='f16': current.load_state_dict({n:v.half().float() if v.ndim>1 else v for n,v in model.state_dict().items()})
    with torch.inference_mode():
        fixture={'dimension':6,'tolerance':1e-4,'texts':[{'text':s,'ids':tok.encode(s).ids,'expected':current.text_vector(tok.encode(s).ids).tolist()} for s in texts],
                 'audio':[{'name':name,'wave':x.tolist(),'expected':current.audio_vector(x).tolist()} for name,x in waves]}
    assert max(abs(a-b) for a,b in zip(fixture['audio'][0]['expected'],fixture['audio'][1]['expected']))>10*fixture['tolerance'], 'amplitude probes must distinguish amplitude normalization'
    for (name,x), c in zip(waves,fixture['audio']):
        assert current.clips(x).shape[1] == (2*len(x) if len(x)<12000 else 24000)
    norm=torch.tensor(fixture['audio'][-1]['expected']).norm().item()
    assert norm<.9999, norm
    final=torch.tensor(fixture['audio'][-1]['expected'])
    assert (torch.nn.functional.normalize(final,dim=0)-final).abs().max().item()>fixture['tolerance']
    with torch.inference_mode():
        clip_vectors=torch.stack([current.audio_vector(c) for c in current.clips(waves[-1][1])])
    weights=torch.tensor([24000,24000,len(waves[-1][1])-48000],dtype=torch.float32)
    weighted=(clip_vectors*weights[:,None]).sum(0)/weights.sum()
    assert (weighted-final).abs().max().item()>fixture['tolerance'], 'fixture must distinguish weighted pooling'
    path=out/f'fixture-{dtype}.json'; path.write_text(json.dumps(fixture))
    target=out/f'model-{dtype}.gguf'
    command=[sys.executable,str(root/'convert_hf_to_gguf.py'),str(out),str(target),'--dtype',dtype,
        '--muq-mulan-revision','2e01c796b71dca71b45251384c04cd7b237c9020',
        '--muq-config',str(out/'muq-config.json'),'--text-config',str(out/'text-config.json')]
    subprocess.run(command,check=True)
    subprocess.run([str(build/'muq-mulan-model-test'),str(target),str(path)],check=True)
    # Native CLI must return joint vectors and reject every unsupported input mode.
    x=waves[-1][1]; data=x.numpy().astype('<f4').tobytes(); wav=out/'input.wav'
    wav.write_bytes(b'RIFF'+struct.pack('<I',36+len(data))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,3,1,24000,96000,4,32)+b'data'+struct.pack('<I',len(data))+data)
    cli=[str(build/'clap-model'),'--model',str(target),'--threads','2']
    for args,expected in [(['--text',texts[0]],fixture['texts'][0]['expected']),(['--audio',str(wav)],fixture['audio'][-1]['expected'])]:
        result=subprocess.run(cli+args,check=True,capture_output=True,text=True)
        error=max(abs(a-b) for a,b in zip(json.loads(result.stdout),expected))
        assert error<fixture['tolerance'],error
        print(dtype,args[0],'CLI error',error,flush=True)
    import numpy as np
    np.save(out/'unsupported-features.npy',np.zeros((2,8),dtype=np.float32))
    rejected=subprocess.run(cli+['--audio-features',str(out/'unsupported-features.npy')],capture_output=True,text=True)
    assert rejected.returncode!=0 and 'raw waveform' in rejected.stderr,rejected
    for flag in (['--frames'],['--clamp3-features'],['--crop-offset','1'],['--crop-offset','0'],['--mert-model',str(target)]):
        result=subprocess.run(cli+['--audio',str(wav)]+flag,capture_output=True,text=True)
        assert result.returncode!=0,(flag,result)
    import torchaudio
    control=torchaudio.functional.resample(x,48000,24000)
    with torch.inference_mode(): control_expected=current.audio_vector(control).tolist()
    control_wav=out/'python-resampled.wav';control_data=control.numpy().astype('<f4').tobytes()
    control_wav.write_bytes(b'RIFF'+struct.pack('<I',36+len(control_data))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,3,1,24000,96000,4,32)+b'data'+struct.pack('<I',len(control_data))+control_data)
    native_source_wav=out/'source-48k.wav'
    native_source_wav.write_bytes(b'RIFF'+struct.pack('<I',36+len(data))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,3,1,48000,192000,4,32)+b'data'+struct.pack('<I',len(data))+data)
    results=[]
    for path in (control_wav,native_source_wav):
        proc=subprocess.run(cli+['--audio',str(path)],check=True,capture_output=True,text=True)
        results.append(json.loads(proc.stdout))
    metrics={'python_resampled_native_error':max(abs(a-b) for a,b in zip(results[0],control_expected)),
             'native_vs_torchaudio_resampler_vector_max_abs':max(abs(a-b) for a,b in zip(*results)),
             'python_resampled_reference':control_expected,'native_on_python_resampled':results[0],
             'native_on_48k_original':results[1]}
    (out/f'resampler-{dtype}.json').write_text(json.dumps(metrics))
    assert metrics['python_resampled_native_error']<fixture['tolerance'],metrics
    print(dtype,'resampler control error',metrics['python_resampled_native_error'],
          'resampler difference',metrics['native_vs_torchaudio_resampler_vector_max_abs'],flush=True)
    print(dtype,'final unnormalized average norm',norm,flush=True)
print('MuLan tiny official-component F32/F16 parity passed')
