#!/usr/bin/env python3
"""Convert local Hugging Face CLAP or CLaMP3 weights to native GGUF.

Conversion is the only step requiring Python and torch. Checkpoints are loaded
with weights_only=True and mmap=True. CLaMP3 symbolic encoder tensors are not
written because native symbolic input is not supported.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import tempfile


def clap_tensor_name(name):
    return (name.replace('audio_model.audio_encoder.', 'audio.')
            .replace('text_model.', 'text.')
            .replace('audio_projection.', 'audio_proj.')
            .replace('text_projection.', 'text_proj.')
            .replace('relative_position_bias_table', 'rel_bias'))


def clamp3_tensor_name(name):
    return name.replace('text_model.', 'text.').replace('audio_model.', 'audio.')


def string(value):
    value = value.encode('utf-8')
    return struct.pack('<Q', len(value)) + value


def load_checkpoint(path, torch):
    try:
        weights = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except TypeError as error:
        raise RuntimeError('torch with safe mmap checkpoint loading is required') from error
    if isinstance(weights, dict) and isinstance(weights.get('model'), dict):
        weights = weights['model']
    if not isinstance(weights, dict):
        raise ValueError('checkpoint does not contain a state dictionary')
    return weights


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


def write_gguf(destination, tensors, metadata, dtype, np, torch):
    encoded = []
    for name, value in tensors:
        if len(name.encode()) >= 64:
            raise ValueError(f'tensor name exceeds ggml limit: {name}')
        half = dtype == 'f16' and value.ndim > 1
        encoded.append((name, value, np.dtype('<f2' if half else '<f4'), 1 if half else 0))
    header = b'GGUF' + struct.pack('<IQQ', 3, len(encoded), len(metadata))
    for key, value in metadata.items():
        header += string(key) + struct.pack('<I', 8) + string(value)
    offset = 0
    for name, value, dt, typ in encoded:
        header += string(name) + struct.pack('<I', value.ndim)
        header += struct.pack('<' + 'Q' * value.ndim, *reversed(value.shape))
        header += struct.pack('<IQ', typ, offset)
        offset += (value.numel() * dt.itemsize + 31) // 32 * 32
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(header)
            output.write(b'\0' * ((-len(header)) % 32))
            for _, value, dt, typ in encoded:
                data = value.detach().to(dtype=torch.float16 if typ else torch.float32).contiguous().numpy()
                data.astype(dt, copy=False).tofile(output)
                output.write(b'\0' * ((-data.nbytes) % 32))
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print(f'{destination}: {len(encoded)} tensors ({dtype}, {metadata["general.architecture"]})')


def convert(source, destination, dtype='f32', clamp3_config=None, tokenizer=None):
    import numpy as np
    import torch
    source, destination = Path(source), Path(destination)
    if clamp3_config is not None or source.is_file():
        tensors, metadata = clamp3_inputs(source, Path(clamp3_config) if clamp3_config else None,
                                          Path(tokenizer) if tokenizer else None, torch)
    else:
        tensors, metadata = clamp_inputs(source, torch)
    write_gguf(destination, tensors, metadata, dtype, np, torch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='CLAP directory or CLaMP3 .pth checkpoint')
    parser.add_argument('destination', type=Path)
    parser.add_argument('--dtype', choices=['f32', 'f16'], default='f32')
    parser.add_argument('--clamp3-config', type=Path,
                        help='CLaMP3 combined config JSON (otherwise infer from sibling XLM-R config)')
    parser.add_argument('--tokenizer', type=Path, help='CLaMP3 tokenizer.json (default: beside checkpoint)')
    args = parser.parse_args()
    convert(args.source, args.destination, args.dtype, args.clamp3_config, args.tokenizer)
