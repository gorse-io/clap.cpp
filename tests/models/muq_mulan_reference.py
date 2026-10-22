"""Reviewed, hash-pinned official components, with constructors/network imports excluded.

Only the named class methods are compiled from the pinned source AST. Real
Transformers XLMRobertaModel and the reviewed MuQ core supply the backbones.
"""
import ast
import hashlib
from pathlib import Path
import sys
import types
import torch
from torch import nn
from einops import rearrange
from muq_reference import InferenceCore

HASHES = {
 'modules/transformer.py': '5eb75b0ce631180fa11abb99f0db35ec1dfe110053026a31092937e1fae76dad',
 'modules/utils.py': 'b457791957f9fcf6c56e4dfabae2d0b1a998b14498b33504eabbe77ff9369c8a',
 'muq_mulan.py': '594843bf29efed147f97768c60946d8e247dce183af085fc5dffcd3d03a7164e',
 'models/audio.py': '05231129cffeb5132aded81b9aabc79ee29fc01513f26879ef98b1b0fbb14b9c',
 'models/text.py': '0cd88116c4fc4bcc310ee87ffaa03f6d3af7f09c593a233ded494e1130c2d0a5',
 'models/mulan.py': 'f91111e8b54b408f749a526f3f13a1d3c1520d2de6692331d55bb35814fbb3fe',
}

def verified(directory, name):
    data = (Path(directory) / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != HASHES[name]:
        raise ValueError(f'unverified MuLan source: {name}: {digest}')
    return data.decode()

def components(directory):
    package = types.ModuleType('pinned_mulan'); package.__path__ = []
    sys.modules[package.__name__] = package
    for name in ('utils', 'transformer'):
        module = types.ModuleType('pinned_mulan.' + name)
        module.__package__ = package.__name__
        sys.modules[module.__name__] = module
        exec(compile(verified(directory, 'modules/' + name + '.py'), name, 'exec'), module.__dict__)
    return sys.modules['pinned_mulan.transformer'].Transformer

def method(directory, file, cls, name):
    tree = ast.parse(verified(directory, file))
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    function = next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.returns = None
    for arg in function.args.args + function.args.kwonlyargs:
        arg.annotation = None
    scope = dict(torch=torch, rearrange=rearrange, exists=lambda x: x is not None, l2norm=lambda x: torch.nn.functional.normalize(x,dim=-1))
    exec(compile(ast.Module(body=[function], type_ignores=[]), file, 'exec'), scope)
    return scope[name]

class Branch(nn.Module):
    def __init__(self, model, adapter, proj):
        super().__init__(); self.model = model; self.transformer = adapter; self.proj = proj

class Joint(nn.Module):
    def __init__(self, cfg, audio_cfg, text_cfg, source, muq_source):
        super().__init__()
        from transformers import XLMRobertaConfig, XLMRobertaModel
        transformer = components(source)
        t, a = cfg['text_transformer'], cfg['audio_transformer']
        def adapter(c):
            return transformer(dim=c['dim'], depth=c['tf_depth'], heads=c['heads'], dim_head=c['dim_head'], ff_mult=c['ff_mult'])
        self.text = Branch(XLMRobertaModel(XLMRobertaConfig(**text_cfg)), adapter(t), nn.Identity())
        self.audio = Branch(InferenceCore(audio_cfg, muq_source), adapter(a), nn.Linear(audio_cfg['encoder_dim'], a['dim']))
        self.text_to_latents = nn.Linear(t['dim'], cfg['mulan']['dim_latent'])
        self.audio_to_latents = nn.Linear(a['dim'], cfg['mulan']['dim_latent'])
        self.cfg, self.source = cfg, source

    def text_vector(self, ids):
        # Execute the official text wrapper forward, including its no-extra-mask
        # custom attention, mean including specials, and identity input projection.
        branch = self.text
        branch.pred_pretrained_model_hidden = lambda **kw: branch.model(**kw).last_hidden_state
        branch.tokenizer = lambda texts, **kw: types.SimpleNamespace(to=lambda device: {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long)})
        branch.__dict__['device'] = torch.device('cpu')
        forward = method(self.source, 'models/text.py', 'TextTransformerPretrained', 'forward')
        latent_holder=types.SimpleNamespace(text=lambda *a,**kw: forward(branch,*a,**kw),
            text_to_latents=self.text_to_latents,norm_type='l2norm')
        norm=method(self.source,'models/mulan.py','MuLanModel','_norm_latents')
        latent_holder._norm_latents=lambda x:norm(latent_holder,x)
        latent=method(self.source,'models/mulan.py','MuLanModel','get_text_latents')
        return latent(latent_holder,raw_texts=['unused'])[0]

    def clips(self, wave):
        wrapper = types.SimpleNamespace(sr=24000, clip_secs=self.cfg['mulan']['clip_secs'])
        return method(self.source, 'muq_mulan.py', 'MuQMuLan', '_get_all_clips')(wrapper, wave)

    def audio_vector(self, wave):
        branch = self.audio
        branch.frozen_pretrained = False; branch.use_layer_idx = -1
        branch._forward_pretrained_model = lambda x: (branch.model(x),)
        # The only additional global used by the official audio forward.
        from contextlib import suppress
        forward = method(self.source, 'models/audio.py', 'AudioSpectrogramTransformerPretrained', 'forward')
        forward.__globals__['suppress'] = suppress
        branch.__dict__['device'] = torch.device('cpu')
        latent_holder=types.SimpleNamespace(audio=lambda *a,**kw: forward(branch,*a,**kw),
            audio_to_latents=self.audio_to_latents,norm_type='l2norm')
        norm=method(self.source,'models/mulan.py','MuLanModel','_norm_latents')
        latent_holder._norm_latents=lambda x:norm(latent_holder,x)
        latent=method(self.source,'models/mulan.py','MuLanModel','get_audio_latents')
        vectors = []
        for clip in self.clips(wave):
            vectors.append(latent(latent_holder,clip[None])[0])
        return torch.stack(vectors).mean(0)
