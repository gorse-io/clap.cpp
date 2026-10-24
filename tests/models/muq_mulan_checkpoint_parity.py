#!/usr/bin/env python3
"""Low-RAM, local-only, published MuLan oracle, one branch per subprocess.

Example: python tests/models/muq_mulan_checkpoint_parity.py --checkpoint artifacts/checkpoint \
 --gguf artifacts/muq-mulan-f16.gguf --dtype f16 --cli artifacts/build/clap \
 --source artifacts/oracle --muq-source /path/to/reviewed/MuQ --out artifacts/published \
 --audio /path/to/music.wav

References/metrics are written BEFORE tolerance assertions. The coordinator uses
only stdlib; all Torch modules exit before native CLI execution. F16 means
storage-rounded parameters with F32 activations, not a half-precision oracle.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

REVISION='2e01c796b71dca71b45251384c04cd7b237c9020'
SHA256='d42ae3f7cb9b66759ee0089ddc70e2f28b130c2d8ba621457358272d32dd0444'
SIZE=2653954401

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','gguf','cli','source','muq-source','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dtype',choices=('f32','f16'),required=True)
    p.add_argument('--audio',type=Path)
    p.add_argument('--text',action='append',help='single-item texts; default includes English/Chinese/empty/Unicode/special')
    p.add_argument('--branch',choices=('text','audio'),help=argparse.SUPPRESS)
    p.add_argument('--tolerance',type=float,default=5e-4)
    return p.parse_args()

def reference(args):
    import types
    import numpy as np
    import torch
    from muq_mulan_reference import Branch, Joint, components
    from muq_reference import InferenceCore, inference_key
    from transformers import XLMRobertaConfig, XLMRobertaModel
    from tokenizers import Tokenizer
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from conversion.base import load_checkpoint
    from conversion.muq_mulan import mulan_inputs
    path,metadata=mulan_inputs(args.checkpoint,REVISION,args.checkpoint/'muq_config.json',args.checkpoint/'text_config.json',args.checkpoint/'tokenizer.json')
    weights=load_checkpoint(path,torch)
    cfg=json.loads(metadata['muq_mulan.config']);audio_cfg=json.loads(metadata['muq.config']);text_cfg=json.loads(metadata['muq_mulan.text_config'])
    transformer=components(args.source)
    selected=args.branch
    c=cfg[selected+'_transformer']
    with torch.device('meta'):
        core=(XLMRobertaModel(XLMRobertaConfig(**text_cfg)) if selected=='text' else InferenceCore(audio_cfg,args.muq_source))
        adapter=transformer(dim=c['dim'],depth=c['tf_depth'],dim_head=c['dim_head'],heads=c['heads'],ff_mult=c['ff_mult'])
        proj=torch.nn.Identity() if selected=='text' else torch.nn.Linear(audio_cfg['encoder_dim'],c['dim'])
        branch=Branch(core,adapter,proj)
        latent=torch.nn.Linear(c['dim'],cfg['mulan']['dim_latent'])
    prefix='mulan.'+selected+'.'
    state={}
    for key,value in weights.items():
        if not key.startswith(prefix):continue
        key=key[len(prefix):]
        if selected=='audio' and key.startswith('model.model.'): key=key.removeprefix('model.')
        if key in branch.state_dict(): state[key]=value
    missing,unexpected=branch.load_state_dict(state,strict=False,assign=True)
    if any(n.startswith(('transformer.','proj.','model.embeddings.','model.encoder.')) or inference_key(n) for n in missing):
        raise ValueError('missing published inference tensors: '+str(missing))
    latent.load_state_dict({k.removeprefix('mulan.'+selected+'_to_latents.'):v for k,v in weights.items() if k.startswith('mulan.'+selected+'_to_latents.')},assign=True)
    # Meta construction also affects nonpersistent buffers, which state_dict
    # cannot restore: XLM-R's ID templates and the adapter's fixed zero/one LN
    # buffers. Materialize only these known buffers, never unused meta weights.
    for module in branch.modules():
        for name in module._non_persistent_buffers_set:
            value=module._buffers.get(name)
            if value is None or not value.is_meta: continue
            if name=='position_ids':
                module._buffers[name]=torch.arange(value.shape[-1],dtype=value.dtype).expand(value.shape)
            elif name in ('token_type_ids','beta'):
                module._buffers[name]=torch.zeros(value.shape,dtype=value.dtype,device='cpu')
            elif name=='gamma':
                module._buffers[name]=torch.ones(value.shape,dtype=value.dtype,device='cpu')
            else:
                raise ValueError('unhandled meta inference buffer: '+name)
    # Round one tensor at a time. All source tensors remain mmap views. The
    # largest text embedding is handled row-wise below to avoid a second copy.
    embedding=None
    if args.dtype=='f16':
        for name,module in branch.named_modules():
            for field,param in list(module.named_parameters(recurse=False)):
                if param.ndim>1:
                    if selected=='text' and name=='model.embeddings.word_embeddings':
                        embedding=module
                        continue
                    setattr(module,field,torch.nn.Parameter(param.detach().half().float(),requires_grad=False))
        for name,param in list(latent.named_parameters()):
            if param.ndim>1: setattr(latent,name,torch.nn.Parameter(param.detach().half().float(),requires_grad=False))
    if embedding is not None:
        # XLM-R uses embedding lookup only; round the selected rows, keeping
        # the 250002x768 source matrix mapped rather than materializing it.
        def embed(ids):
            return torch.nn.functional.embedding(ids,embedding.weight).half().float()
        embedding.forward=embed
    branch.eval();latent.eval()
    holder=types.SimpleNamespace(cfg=cfg,source=args.source)
    setattr(holder,selected,branch);setattr(holder,selected+'_to_latents',latent)
    if selected=='text':
        tok=Tokenizer.from_file(str(args.checkpoint/'tokenizer.json'))
        texts=args.text or ['classical genres, hopeful mood, piano.','一首适合海边风景的小提琴曲，节奏欢快','', 'e\u0301 音乐', 'a <mask> b']
        cases=[]
        with torch.inference_mode():
            for index,text in enumerate(texts):
                ids=tok.encode(text).ids
                vector=Joint.text_vector(holder,ids).numpy()
                np.save(args.out/f'text-{index}-reference.npy',vector)
                cases.append({'text':text,'ids':ids,'expected':vector.tolist(),'norm':float(np.linalg.norm(vector))})
        (args.out/'text-reference.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    else:
        import torchaudio
        import struct
        if args.audio:
            wave,sr=torchaudio.load(args.audio);wave=wave.mean(0)
            if sr!=24000: wave=torchaudio.functional.resample(wave,sr,24000)
        else:
            t=torch.arange(24000*11+137)/24000
            wave=.17*torch.sin(2*torch.pi*220*t)*(0.1+t/8)+.08*torch.cos(2*torch.pi*733*t)
            # Keep deterministic broadband energy away from ill-conditioned
            # log-mel FFT noise-floor differences (no amplitude normalization).
            wave+=.002*torch.randn(wave.shape,generator=torch.Generator().manual_seed(20261022))
        holder.clips=lambda x:Joint.clips(holder,x)
        with torch.inference_mode():vector=Joint.audio_vector(holder,wave).numpy()
        np.save(args.out/'audio-reference.npy',vector);np.save(args.out/'audio-waveform.npy',wave.numpy())
        data=wave.numpy().astype('<f4').tobytes();wav=args.out/'audio-24k.wav'
        wav.write_bytes(b'RIFF'+struct.pack('<I',36+len(data))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,3,1,24000,96000,4,32)+b'data'+struct.pack('<I',len(data))+data)
        (args.out/'audio-reference.json').write_text(json.dumps({'wave':str(wav),'samples':len(wave),'expected':vector.tolist(),'norm':float(np.linalg.norm(vector))},indent=2))

def comparison_metrics(actual, expected):
    if len(actual) != len(expected) or not actual:
        raise ValueError('oracle vector dimension mismatch/empty vector')
    if not all(math.isfinite(value) for vector in (actual, expected) for value in vector):
        raise ValueError('non-finite native/reference oracle component')
    error=max(abs(a-b) for a,b in zip(actual,expected))
    dot=sum(a*b for a,b in zip(actual,expected));na=sum(a*a for a in actual)**.5;nb=sum(b*b for b in expected)**.5
    return {'max_abs_error':error,'cosine':dot/(na*nb) if na*nb else 0,
            'native_norm':na,'reference_norm':nb,'dimension':len(actual)}


def main():
    args=parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.branch:
        reference(args);return
    checkpoint=args.checkpoint/'pytorch_model.bin'
    if not checkpoint.is_file() or checkpoint.stat().st_size!=SIZE:
        raise ValueError('complete published pytorch_model.bin is required; partial files are never read')
    digest=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
    if digest!=SHA256:raise ValueError('published checkpoint SHA256 mismatch')
    # Child modules and tensors are gone before starting the native process.
    for branch in ('text','audio'):
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--branch',branch],check=True)
        cases=json.loads((args.out/(branch+'-reference.json')).read_text())
        if branch=='audio':cases=[cases]
        metrics=[]
        for index,case in enumerate(cases):
            command=[str(args.cli), 'embedding','--model',str(args.gguf),'--threads','2']
            command+=['--text',case['text']] if branch=='text' else ['--audio',case['wave']]
            proc=subprocess.run(command,check=True,capture_output=True,text=True)
            actual=json.loads(proc.stdout);expected=case['expected']
            (args.out/f'{branch}-{index}-native.json').write_text(json.dumps(actual))
            metrics.append({'case':index,**comparison_metrics(actual,expected)})
        if branch=='audio' and args.audio:
            # Characterize native decode/resampling separately; no claim of
            # bit equivalence to torchaudio/librosa resamplers.
            original=subprocess.run([str(args.cli), 'embedding','--model',str(args.gguf),'--threads','2','--audio',str(args.audio)],check=True,capture_output=True,text=True)
            vec=json.loads(original.stdout);control=json.loads((args.out/'audio-0-native.json').read_text())
            (args.out/'audio-original-native.json').write_text(json.dumps(vec))
            metrics.append({'resampler_control_max_abs':comparison_metrics(vec,control)['max_abs_error']})
        (args.out/(branch+'-metrics.json')).write_text(json.dumps(metrics,indent=2))
        print(branch,metrics,flush=True)
        assert all(m.get('dimension',512)==512 and m.get('max_abs_error',0)<=args.tolerance for m in metrics), metrics

if __name__=='__main__':main()
