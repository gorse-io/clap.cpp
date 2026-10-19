"""Independent full published SAAS checkpoint oracle (not tiny/random weights)."""
from pathlib import Path
import argparse,gc,json,subprocess,sys
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import BertConfig,BertModel,XLMRobertaConfig,XLMRobertaModel
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('build', type=Path)
parser.add_argument('checkpoint', type=Path, help='Directory with one official .pth, XLM-R config.json and tokenizer.json')
parser.add_argument('output', type=Path)
parser.add_argument('--dtype', choices=['f32','f16'], default='f32')
parser.add_argument('--gguf', type=Path, help='Verify an already converted GGUF instead of converting again')
args=parser.parse_args()
root=args.checkpoint
out=args.output;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
paths=list(root.glob('*.pth'))
if len(paths)!=1: raise ValueError('Checkpoint directory must contain exactly one .pth')
checkpoint_path=paths[0]
checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
state=checkpoint['model']
tok=Tokenizer.from_file(str(root/'tokenizer.json'))
texts=['A calm acoustic guitar melody.','轻快的钢琴和小提琴','موسيقى هادئة','café e\u0301 ＡＢＣ ①','a\u200bb\u00adc\ufeffd','a <mask> b','🎻🎶 \U0001fae8\U0001fae9']
tokenizer_cases=[{'text':t,'ids':tok.encode(t).ids} for t in texts]

def segments(data,pad):
    length=len(data)
    chunks=[data[i:i+128] for i in range(0,length,128)]
    chunks[-1]=data[-128:]
    for i,chunk in enumerate(chunks):
        n=len(chunk)
        full=torch.full((128,)+tuple(data.shape[1:]),pad,dtype=data.dtype)
        full[:n]=chunk
        mask=torch.zeros(128,dtype=torch.int64);mask[:n]=1
        yield full,mask,min(128,length-i*128)

def load_model(cls,config,prefix):
    with torch.device('meta'):
        model=cls(config)
    subset={k[len(prefix)+1:]:v for k,v in state.items() if k.startswith(prefix+'.')}
    model.load_state_dict(subset,assign=True,strict=True)
    # HF nonpersistent position/token-type buffers were created on meta and
    # are not in state_dict: materialize them explicitly before inference.
    embeddings=model.embeddings
    embeddings.register_buffer('position_ids',torch.arange(config.max_position_embeddings).expand((1,-1)),persistent=False)
    embeddings.register_buffer('token_type_ids',torch.zeros((1,config.max_position_embeddings),dtype=torch.long),persistent=False)
    return model.eval()

def projection(x,prefix):
    return torch.nn.functional.linear(x,state[prefix+'.weight'],state[prefix+'.bias'])

@torch.no_grad()
def pooled(model,data,is_audio):
    result=[];weights=[]
    for x,mask,w in segments(data,0 if is_audio else 1):
        args={'inputs_embeds':x[None]} if is_audio else {'input_ids':x[None]}
        y=model(**args,attention_mask=mask[None]).last_hidden_state
        y=(y*mask[None,:,None]).sum(1)/mask.sum()
        result.append(projection(y,'audio_proj' if is_audio else 'text_proj')[0]);weights.append(w)
    weights=torch.tensor(weights,dtype=torch.float32)
    return (torch.stack(result)*weights[:,None]).sum(0)/weights.sum()

fixture={'projection_dim':768,'embedding_tolerance':0.0005 if args.dtype=='f32' else 0.005,'tokenizer_cases':tokenizer_cases,'text_cases':[],'audio_cases':[]}
config=XLMRobertaConfig.from_json_file(root/'config.json')
text=load_model(XLMRobertaModel,config,'text_model')
for title,ids in [('published SAAS English',tokenizer_cases[0]['ids']),('published SAAS Chinese',tokenizer_cases[1]['ids']),('published SAAS overlapping long text',tok.encode('Gentle piano music with warm strings. '*25).ids)]:
    y=pooled(text,torch.tensor(ids),False)
    assert torch.isfinite(y).all(),title
    fixture['text_cases'].append({'name':title,'ids':ids,'expected':y.tolist()})
    print(title,len(ids),y[:4].tolist(),flush=True)
del text;gc.collect()
config=BertConfig(vocab_size=1,hidden_size=768,num_hidden_layers=12,num_attention_heads=12,intermediate_size=3072,max_position_embeddings=128)
audio=load_model(BertModel,config,'audio_model')
for title,n in [('published SAAS short MERT features',5),('published SAAS overlapping long MERT features',131)]:
    data=torch.sin(torch.arange(n*768,dtype=torch.float32)/113).reshape(n,768)*0.2
    y=pooled(audio,torch.cat([torch.zeros(1,768),data,torch.zeros(1,768)]),True)
    assert torch.isfinite(y).all(),title
    fixture['audio_cases'].append({'name':title,'frames':n,'features':data.flatten().tolist(),'expected':y.tolist()})
    if n==5:np.save(out/'short-audio.npy',data.numpy()[None])
    print(title,y[:4].tolist(),flush=True)
(out/'fixture.json').write_text(json.dumps(fixture))
print('Saved published-checkpoint oracle',out/'fixture.json',flush=True)

del audio,state,checkpoint
gc.collect()
converter=Path(__file__).resolve().parents[2]/'convert_hf_to_gguf.py'
gguf=args.gguf or out/('model-'+args.dtype+'.gguf')
if args.gguf is None:
    subprocess.run([sys.executable,str(converter),str(checkpoint_path),str(gguf),'--dtype',args.dtype],check=True)
subprocess.run([str(args.build/'clamp3-model-test'),str(gguf),str(out/'fixture.json')],check=True)
print('Published CLaMP3 checkpoint parity passed:',args.dtype)
