"""CLAP configuration, tokenizer and tensor conversion."""
import json

from .base import ModelBase, load_checkpoint, write_gguf


def clap_tensor_name(name):
    return (name.replace('audio_model.audio_encoder.', 'audio.')
            .replace('text_model.', 'text.')
            .replace('audio_projection.', 'audio_proj.')
            .replace('text_projection.', 'text_proj.')
            .replace('relative_position_bias_table', 'rel_bias'))


def clamp_inputs(source, torch):
    from transformers import ClapConfig
    if not source.is_dir():
        raise ValueError('CLAP source must be a checkpoint directory')
    raw = json.loads((source / 'config.json').read_text())
    if raw.get('model_type') != 'clap':
        raise ValueError('expected model_type clap')
    config = ClapConfig.from_dict(raw).to_dict()
    a, t = config['audio_config'], config['text_config']
    if a['enable_fusion']:
        raise ValueError('fusion models are not supported; use the unfused larger_clap_music checkpoint')
    if (a['patch_size'] != 4 or a['patch_stride'] != [4, 4]
            or not a['enable_patch_layer_norm'] or not a['flatten_patch_embeds']
            or a['patch_embed_input_channels'] != 1 or not a['qkv_bias']):
        raise ValueError('unsupported audio patch embedding configuration')
    if a['hidden_act'] != 'gelu' or t['hidden_act'] != 'gelu':
        raise ValueError('only exact GELU encoders are supported')
    if any(x['projection_hidden_act'] != 'relu' for x in (a, t)):
        raise ValueError('only ReLU projections are supported')
    if t.get('position_embedding_type', 'absolute') != 'absolute':
        raise ValueError('only absolute text positions are supported')
    tokenizer = json.loads((source / 'tokenizer.json').read_text())
    if tokenizer['model'].get('type') != 'BPE':
        raise ValueError('expected RoBERTa byte-level BPE tokenizer')
    preprocessor = source / 'preprocessor_config.json'
    if preprocessor.exists():
        p = json.loads(preprocessor.read_text())
        required = dict(sampling_rate=48000, feature_size=64, hop_length=480,
                        fft_window_size=1024, frequency_min=50, frequency_max=14000,
                        padding='repeatpad', truncation='rand_trunc')
        for key, value in required.items():
            if key in p and p[key] != value:
                raise ValueError(f'unsupported preprocessor {key}: {p[key]}')
        if p.get('top_db') is not None:
            raise ValueError('top_db must be null')
    weights = load_checkpoint(source / 'pytorch_model.bin', torch)
    selected = []
    for name, value in weights.items():
        if not value.is_floating_point():
            continue
        name = clap_tensor_name(name)
        if not name.startswith(('audio.', 'text.', 'audio_proj.', 'text_proj.', 'logit_scale')):
            raise ValueError(f'unrecognized tensor: {name}')
        selected.append((name, value))
    metadata = {'general.architecture': 'clap', 'general.name': 'laion/larger_clap_music',
                'clap.config': json.dumps(config, separators=(',', ':')),
                'clap.tokenizer': json.dumps(tokenizer, ensure_ascii=False, separators=(',', ':'))}
    return selected, metadata


@ModelBase.register('clap')
class ClapModel(ModelBase):
    def write(self):
        import numpy as np
        import torch
        tensors, metadata = clamp_inputs(self.source, torch)
        write_gguf(self.destination, tensors, metadata, self.dtype, np, torch)
