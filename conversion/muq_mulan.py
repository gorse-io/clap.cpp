"""Local-only OpenMuQ/MuQ-MuLan conversion; no model source is executed."""
import json
import math
from pathlib import Path

from .base import ModelBase, load_checkpoint, write_gguf

SOURCE_REVISION = '28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a'


def validate_config(cfg, audio, text):
    def positive(value, name, integer=True):
        if (type(value) is not int if integer else type(value) not in (int, float)) or not math.isfinite(value) or value <= 0 or (integer and value > 10000000):
            raise ValueError('invalid MuLan ' + name)
    m, a, t = cfg['mulan'], cfg['audio_transformer'], cfg['text_transformer']
    positive(m['clip_secs'], 'clip_secs')
    positive(m['dim_latent'], 'dim_latent')
    if type(m['sr']) is not int or m['sr'] != 24000 or m['clip_secs'] > 3600 or m.get('hierarchical_contrastive_loss', False):
        raise ValueError('unsupported MuLan sample rate/hierarchical variant')
    if cfg['audio_model']['name'] != 'OpenMuQ/MuQ-large-msd-iter' or type(cfg['audio_model']['use_layer_idx']) is not int or cfg['audio_model']['use_layer_idx'] != -1:
        raise ValueError('MuLan supports only final-layer OpenMuQ/MuQ-large-msd-iter')
    if type(cfg['audio_model']['model_dim']) is not int or cfg['audio_model']['model_dim'] != audio['encoder_dim'] or a['tf_depth'] != 0:
        raise ValueError('MuLan requires matching MuQ dimension and depth0 audio adapter')
    if cfg['text_model']['name'] != 'xlm-roberta-base' or (cfg['text_model'].get('model_dim') is not None and (type(cfg['text_model']['model_dim']) is not int or cfg['text_model']['model_dim'] != text['hidden_size'])):
        raise ValueError('MuLan supports only identity-projected xlm-roberta-base text')
    if t['dim'] != text['hidden_size']:
        raise ValueError('MuLan text adapter dimension mismatch')
    for branch in (a, t):
        for key in ('dim', 'heads', 'dim_head'):
            positive(branch[key], key)
        positive(branch['ff_mult'], 'ff_mult', False)
        if branch['ff_mult'] > 100 or branch['heads'] * branch['dim_head'] > 1048576 or not 1 <= int(branch['dim'] * branch['ff_mult'] * 2 / 3) <= 1048576:
            raise ValueError('invalid MuLan adapter geometry')
        if type(branch['tf_depth']) is not int or not 0 <= branch['tf_depth'] <= 1000:
            raise ValueError('invalid MuLan tf_depth')
    if text.get('model_type') != 'xlm-roberta' or text.get('hidden_act') != 'gelu' or text.get('position_embedding_type', 'absolute') != 'absolute' or text.get('is_decoder', False) or text.get('add_cross_attention', False):
        raise ValueError('unsupported MuLan XLM-R architecture')
    for key in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'vocab_size', 'max_position_embeddings', 'type_vocab_size'):
        positive(text[key], key)
    positive(text['layer_norm_eps'], 'layer_norm_eps', False)
    if text['hidden_size'] % text['num_attention_heads'] or text['max_position_embeddings'] <= text['pad_token_id'] + 2:
        raise ValueError('invalid MuLan XLM-R geometry')
    if any(type(text[k]) is not int for k in ('bos_token_id','pad_token_id','eos_token_id')) or [text[k] for k in ('bos_token_id','pad_token_id','eos_token_id')] != [0,1,2] or text['vocab_size'] < 4:
        raise ValueError('MuLan requires XLM-R special IDs 0/1/2/3')


def mulan_inputs(source, revision, muq_config, text_config, tokenizer):
    if not revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('MuLan conversion requires --muq-mulan-revision (full lowercase commit SHA)')
    def local(option, default, label):
        path = Path(option) if option else source / default
        if not path.is_file():
            raise ValueError('MuLan requires local --' + label + ': ' + str(path))
        return json.loads(path.read_text())
    cfg = local(None, 'config.json', 'config')
    audio = local(muq_config, 'muq-config.json', 'muq-config')
    text = local(text_config, 'text-config.json', 'text-config')
    tok = local(tokenizer, 'tokenizer.json', 'tokenizer')
    validate_config(cfg, audio, text)
    # Use the same independently validated MuQ config contract, without loading
    # a model, importing Torch, or accessing the network.
    from .muq import muq_inputs
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary); (path / 'config.json').write_text(json.dumps(audio))
        muq_inputs(path, revision)
    if tok.get('model', {}).get('type', 'Unigram') != 'Unigram' or len(tok['model']['vocab']) != text['vocab_size']:
        raise ValueError('MuLan tokenizer vocabulary disagrees with local XLM-R config')
    if [entry[0] for entry in tok['model']['vocab'][:4]] != ['<s>', '<pad>', '</s>', '<unk>']:
        raise ValueError('MuLan requires XLM-R tokenizer special IDs')
    metadata = {'general.architecture':'muq_mulan', 'general.name':'OpenMuQ/MuQ-MuLan-large',
        'general.license':'cc-by-nc-4.0', 'muq_mulan.source_license':'MIT',
        'muq_mulan.source_revision':SOURCE_REVISION, 'muq_mulan.model_revision':revision,
        'muq_mulan.config':json.dumps(cfg,separators=(',',':')),
        'muq.config':json.dumps(audio,separators=(',',':')),
        'muq_mulan.text_config':json.dumps(text,separators=(',',':')),
        'muq_mulan.tokenizer':json.dumps(tok,ensure_ascii=False,separators=(',',':'))}
    return source / 'pytorch_model.bin', metadata


def tensor_name(name):
    """Return inference name or None for explicitly known unused tensors."""
    audio = 'mulan.audio.model.model.'
    if name.startswith(audio):
        short = name[len(audio):]
        if short.endswith('.num_batches_tracked'):
            if short.startswith(('conv.', 'conformer.layers.')): return None
        if short.startswith(('conv.', 'conformer.layers.', 'conformer.layer_norm.')): return short
        if short.startswith(('rvq.', 'linear.', 'conformer.pos_conv_embed.')) or short == 'cls_token': return None
        if short == 'conformer.embed_positions.inv_freq': return None  # checked separately
    if name.startswith(('mulan.audio.aggregator.', 'mulan.text.model.pooler.', 'mulan.contrast.')):
        return None
    for original, target in [('mulan.text.model.embeddings.', 'text.embeddings.'),
            ('mulan.text.model.encoder.', 'text.encoder.'), ('mulan.text.transformer.', 'adapter.'),
            ('mulan.audio.proj.', 'audio_proj.'), ('mulan.text_to_latents.', 'text_proj.'),
            ('mulan.audio_to_latents.', 'audio_latent.')]:
        if name.startswith(original): return target + name[len(original):]
    raise ValueError('unrecognized MuLan tensor: ' + name)


def expected_shapes(cfg, audio, text):
    """Complete inference tensor schema in PyTorch order (including buffers)."""
    shapes = {'frontend.window': (2048,), 'frontend.filters': (1025,128)}
    def linear(name, source, target, bias=True):
        shapes[name+'.weight']=(target,source)
        if bias: shapes[name+'.bias']=(target,)
    def norm(name, dim):
        shapes[name+'.weight']=(dim,); shapes[name+'.bias']=(dim,)
    def batch(name, dim):
        norm(name,dim); shapes[name+'.running_mean']=(dim,); shapes[name+'.running_var']=(dim,)
    d=text['hidden_size']; h=audio['encoder_dim']; c=audio['conv_dim']
    for name,size in [('word',text['vocab_size']),('position',text['max_position_embeddings']),('token_type',text['type_vocab_size'])]:
        shapes['text.embeddings.'+name+'_embeddings.weight']=(size,d)
    norm('text.embeddings.LayerNorm',d)
    for i in range(text['num_hidden_layers']):
        p=f'text.encoder.layer.{i}'
        for name in ('query','key','value'): linear(p+'.attention.self.'+name,d,d)
        linear(p+'.attention.output.dense',d,d); norm(p+'.attention.output.LayerNorm',d)
        linear(p+'.intermediate.dense',d,text['intermediate_size'])
        linear(p+'.output.dense',text['intermediate_size'],d); norm(p+'.output.LayerNorm',d)
    t=cfg['text_transformer']; inner=t['heads']*t['dim_head']; ff=int(d*t['ff_mult']*2/3)
    for i in range(t['tf_depth']):
        p=f'adapter.layers.{i}'
        shapes[p+'.0.norm.learned_gamma']=(d,)
        shapes[p+'.0.q_scale']=(t['dim_head'],); shapes[p+'.0.k_scale']=(t['dim_head'],)
        linear(p+'.0.to_q',d,inner,False);linear(p+'.0.to_kv',d,2*inner,False);linear(p+'.0.to_out.0',inner,d,False)
        shapes[p+'.1.0.learned_gamma']=(d,)
        linear(p+'.1.1',d,2*ff,False);linear(p+'.1.4',ff,d,False)
    linear('audio_proj',h,cfg['audio_transformer']['dim'])
    linear('audio_latent',cfg['audio_transformer']['dim'],cfg['mulan']['dim_latent'])
    linear('text_proj',d,cfg['mulan']['dim_latent'])
    for stage in range(2):
        for layer in range(1,4):
            p=f'conv.conv.{stage}'; inputs=1 if stage==0 and layer!=2 else c
            shapes[p+f'.conv{layer}.weight']=(c,inputs,3,3);shapes[p+f'.conv{layer}.bias']=(c,)
            batch(p+f'.bn{layer}',c)
    linear('conv.linear',c*32,h); norm('conformer.layer_norm',h)
    ec=audio['w2v2_config']
    for i in range(audio['encoder_depth']):
        p=f'conformer.layers.{i}'
        for name in ('ffn1_layer_norm','ffn2_layer_norm','self_attn_layer_norm','final_layer_norm'):
            norm(p+'.'+name,h)
        for name in ('ffn1','ffn2'):
            linear(p+'.'+name+'.intermediate_dense',h,ec['intermediate_size'])
            linear(p+'.'+name+'.output_dense',ec['intermediate_size'],h)
        for name in ('linear_q','linear_k','linear_v','linear_out'): linear(p+'.self_attn.'+name,h,h)
        norm(p+'.conv_module.layer_norm',h)
        shapes[p+'.conv_module.pointwise_conv1.weight']=(2*h,h,1)
        shapes[p+'.conv_module.depthwise_conv.weight']=(h,1,ec['conv_depthwise_kernel_size'])
        batch(p+'.conv_module.batch_norm',h)
        shapes[p+'.conv_module.pointwise_conv2.weight']=(h,h,1)
    return shapes


@ModelBase.register('muq_mulan')
class MuQMuLanModel(ModelBase):
    def write(self):
        checkpoint, metadata = mulan_inputs(self.source, self.options.get('muq_mulan_revision'),
            self.options.get('muq_config'), self.options.get('text_config'), self.options.get('tokenizer'))
        if not checkpoint.is_file():
            raise ValueError('MuLan requires complete local pytorch_model.bin: ' + str(checkpoint))
        import numpy as np
        import torch
        import torchaudio
        weights = load_checkpoint(checkpoint, torch)
        tensors = []
        shapes = expected_shapes(json.loads(metadata['muq_mulan.config']), json.loads(metadata['muq.config']), json.loads(metadata['muq_mulan.text_config']))
        for original, value in weights.items():
            name = tensor_name(original)
            if original == 'mulan.audio.model.model.conformer.embed_positions.inv_freq':
                raw=json.loads(metadata['muq.config']); head=raw['encoder_dim']//raw['w2v2_config']['num_attention_heads']
                expected=1. / (raw['w2v2_config']['rotary_embedding_base'] ** (torch.arange(0,head,2).float()/head))
                if value.shape != expected.shape or not torch.allclose(value,expected,rtol=1e-6,atol=0):
                    raise ValueError('MuLan MuQ rotary frequencies disagree with config')
            if name is None: continue
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                raise ValueError('MuLan source inference weights must be F32: ' + original)
            if name not in shapes or tuple(value.shape) != shapes[name]:
                raise ValueError('wrong MuLan tensor shape/name: ' + original)
            if name.endswith('.running_var') and (value < 0).any():
                raise ValueError('negative MuLan batch variance: ' + original)
            flat=value.reshape(-1)
            for offset in range(0,flat.numel(),1048576):
                chunk = flat[offset:offset+1048576]
                if not torch.isfinite(chunk).all():
                    raise ValueError('non-finite MuLan tensor: ' + original)
                if self.dtype == 'f16' and value.ndim > 1 and not torch.isfinite(chunk.to(dtype=torch.float16)).all():
                    raise ValueError('MuLan tensor overflows F16 storage: ' + original)
            tensors.append((name,value))
        missing=set(shapes).difference(n for n,_ in tensors).difference({'frontend.window','frontend.filters'})
        if missing: raise ValueError('missing MuLan tensors: ' + ', '.join(sorted(missing)))
        frontend=torchaudio.transforms.MelSpectrogram(sample_rate=24000,n_fft=2048,hop_length=240,n_mels=128)
        tensors += [('frontend.window',frontend.spectrogram.window),('frontend.filters',frontend.mel_scale.fb)]
        write_gguf(self.destination,tensors,metadata,self.dtype,np,torch,
                   keep_f32=lambda name,value: name.startswith('frontend.'))
