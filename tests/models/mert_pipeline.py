#!/usr/bin/env python3
"""Exercise native MERT -> CLaMP3 and feature-file equivalence.
Synthetic CLaMP3 weights establish pipeline plumbing, not published retrieval quality.
"""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from transformers import BertConfig, BertModel
from mert_cache import install
from feature_input import encode

build,out=map(Path,sys.argv[1:3]); root=Path(__file__).resolve().parents[2]
source=out/'source'; config=json.loads((source/'config.json').read_text()); hidden=config['hidden_size']
torch.manual_seed(12);torch.set_num_threads(2)
fixture=out/'pipeline';fixture.mkdir(exist_ok=True)
install(out/'hf-cache',source)
a=BertConfig(hidden_size=hidden,intermediate_size=hidden*2,num_attention_heads=2,num_hidden_layers=1,
             vocab_size=1,max_position_embeddings=128,type_vocab_size=2,pad_token_id=0,
             hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
t=json.loads((build/'clamp3-fixture/clamp3_config.json').read_text())['text_config']
# Keep the existing verified text branch and tokenizer; only audio width changes.
checkpoint=torch.load(build/'clamp3-fixture/tiny-clamp3.pth',weights_only=True)['model']
checkpoint={k:v for k,v in checkpoint.items() if not k.startswith(('audio_model.','audio_proj.'))}
audio=BertModel(a).eval();proj=torch.nn.Linear(hidden,6).eval()
checkpoint.update({'audio_model.'+k:v for k,v in audio.state_dict().items()})
checkpoint.update({'audio_proj.'+k:v for k,v in proj.state_dict().items()})
torch.save({'model':checkpoint},fixture/'model.pth')
c=dict(model_type='clamp3',variant='synthetic-pipeline',projection_dim=6,chunk_length=128,
       audio_feature_dim=hidden,audio_config=a.to_dict(),text_config=t)
(fixture/'config.json').write_text(json.dumps(c))
subprocess.run([sys.executable,str(root/'convert_hf_to_gguf.py'),str(fixture/'model.pth'),str(fixture/'model.gguf'),
               '--clamp3-config',str(fixture/'config.json'),'--tokenizer',str(build/'clamp3-fixture/tokenizer.json')],check=True)
cli=str(build/'clap')
for dtype in ['f32']:
    mert=str(out/f'model-{dtype}.gguf')
    for case in ['broadband','two_chunks','discard_tail','resampled-input']:
        wave=out/f'{case}.wav'
        data=json.loads(subprocess.check_output([cli, 'embedding','-m',mert,'-a',str(wave),'--clamp3-features','-t','2']))
        features=np.asarray(data['values'],dtype='float32').reshape(data['frames'],hidden)
        expected=encode(build,fixture/'model.gguf',features.ravel(),data['frames'],fixture)
        actual=json.loads(subprocess.check_output([cli, 'embedding','-m',str(fixture/'model.gguf'),
                                                  '-a',str(wave),'-t','2']))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
        with torch.no_grad():
            x=torch.cat([torch.zeros(1,hidden),torch.from_numpy(features),torch.zeros(1,hidden)])[None]
            mask=torch.zeros(1,128,dtype=torch.long);mask[:,:x.shape[1]]=1
            x=torch.nn.functional.pad(x,(0,0,0,128-x.shape[1]))
            y=audio(inputs_embeds=x,attention_mask=mask).last_hidden_state
            y=proj((y*mask[:,:,None]).sum(1)/mask.sum(1)[:,None])[0].numpy()
        np.testing.assert_allclose(actual,y,rtol=1e-4,atol=1e-5)
        print(dtype,case,'native audio -> MERT -> CLaMP3 max_abs',
              float(np.max(np.abs(np.asarray(actual)-y))),flush=True)

wave = str(out/'broadband.wav')
mert = str(out/'model-f32.gguf')
clamp = str(fixture/'model.gguf')
for arguments in [
    ['-m',mert,'-a',wave,'--clamp3-features','--frames'],
    ['-m',mert,'-a',wave,'--frames','--crop-offset','0'],
    ['-m',mert,'-a',wave,'--clamp3-features','--crop-offset','0'],
    ['-m',clamp,'-a',wave,'--crop-offset','0'],
    ['-m',clamp,'-a',wave,'--audio-features','obsolete.npy'],
    ['-m',clamp,'-a',wave,'--mert-model',mert],
]:
    result = subprocess.run([cli, 'embedding',*arguments], capture_output=True, text=True)
    assert result.returncode > 0 and result.stderr, (arguments,result.returncode,result.stderr[:1000])
print('Native pipeline and CLI validation passed')
