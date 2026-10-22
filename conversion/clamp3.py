"""CLaMP3 configuration inference and text/audio tensor conversion."""
import json
from pathlib import Path

from .base import ModelBase, load_checkpoint, write_gguf


def clamp3_tensor_name(name):
    return name.replace('text_model.', 'text.').replace('audio_model.', 'audio.')


def inferred_clamp3_config(text, weights):
    audio_hidden = int(weights['audio_model.embeddings.position_embeddings.weight'].shape[1])
    projection = int(weights['audio_proj.weight'].shape[0])
    audio_layers = 1 + max(int(name.split('.')[3]) for name in weights
                           if name.startswith('audio_model.encoder.layer.'))
    return {
        'model_type': 'clamp3', 'variant': 'saas', 'projection_dim': projection,
        'chunk_length': 128, 'audio_feature_dim': audio_hidden,
        'text_config': text,
        'audio_config': {
            'hidden_size': audio_hidden, 'num_hidden_layers': audio_layers,
            'num_attention_heads': max(1, audio_hidden // 64),
            'intermediate_size': audio_hidden * 4, 'max_position_embeddings': 128,
            'type_vocab_size': 2, 'vocab_size': 1, 'pad_token_id': 0,
            'hidden_act': 'gelu', 'layer_norm_eps': 1e-12,
            'position_embedding_type': 'absolute'
        }
    }


def clamp3_inputs(source, config_path, tokenizer_path, torch):
    checkpoint = source if source.is_file() else next(iter(sorted(source.glob('*.pth'))), None)
    if checkpoint is None:
        raise ValueError('CLaMP3 source contains no .pth checkpoint')
    weights = load_checkpoint(checkpoint, torch)
    if config_path:
        config = json.loads(config_path.read_text())
    else:
        text_path = checkpoint.parent / 'config.json'
        if not text_path.exists():
            raise ValueError('CLaMP3 conversion requires --clamp3-config or sibling config.json')
        config = inferred_clamp3_config(json.loads(text_path.read_text()), weights)
    if config.get('model_type') != 'clamp3':
        raise ValueError('expected model_type clamp3')
    if config.get('chunk_length') != 128:
        raise ValueError('CLaMP3 chunk_length must be 128')
    tokenizer_file = tokenizer_path or checkpoint.parent / 'tokenizer.json'
    tokenizer = json.loads(tokenizer_file.read_text())
    selected = []
    allowed = ('text_model.embeddings.', 'text_model.encoder.', 'text_proj.',
               'audio_model.embeddings.position_embeddings.',
               'audio_model.embeddings.token_type_embeddings.',
               'audio_model.embeddings.LayerNorm.', 'audio_model.encoder.', 'audio_proj.')
    for name, value in weights.items():
        if name.startswith('symbolic_') or not name.startswith(allowed) or not value.is_floating_point():
            continue
        selected.append((clamp3_tensor_name(name), value))
    required = {'text.embeddings.word_embeddings.weight', 'audio.embeddings.position_embeddings.weight',
                'text_proj.weight', 'text_proj.bias', 'audio_proj.weight', 'audio_proj.bias'}
    missing = required.difference(name for name, _ in selected)
    if missing:
        raise ValueError('missing CLaMP3 tensors: ' + ', '.join(sorted(missing)))
    metadata = {'general.architecture': 'clamp3', 'general.name': checkpoint.stem,
                'clamp3.config': json.dumps(config, separators=(',', ':')),
                'clamp3.tokenizer': json.dumps(tokenizer, ensure_ascii=False, separators=(',', ':'))}
    return selected, metadata


@ModelBase.register('clamp3')
class Clamp3Model(ModelBase):
    def write(self):
        import numpy as np
        import torch
        config = self.options.get('clamp3_config')
        tokenizer = self.options.get('tokenizer')
        tensors, metadata = clamp3_inputs(self.source, Path(config) if config else None,
                                          Path(tokenizer) if tokenizer else None, torch)
        from .mert import mert_inputs, resolve_mert_source
        mert_source, revision = resolve_mert_source(self.options.get('mert_revision'))
        frontend, frontend_metadata = mert_inputs(mert_source, revision, torch)
        config = json.loads(metadata['clamp3.config'])
        mert_config = json.loads(frontend_metadata['mert.config'])
        if (type(config.get('audio_feature_dim')) is not int or
            config['audio_feature_dim'] != mert_config['hidden_size'] or
            config['audio_config']['hidden_size'] != mert_config['hidden_size']):
            raise ValueError('CLaMP3/MERT feature dimension mismatch')
        tensors.extend(('mert.' + name, value) for name, value in frontend)
        metadata.update({'clamp3.audio_frontend': 'mert-v1',
                         'clamp3.mert.config': frontend_metadata['mert.config'],
                         'clamp3.mert.source_revision': frontend_metadata['mert.source_revision']})
        write_gguf(self.destination, tensors, metadata, self.dtype, np, torch)
