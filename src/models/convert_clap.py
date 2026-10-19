#!/usr/bin/env python3
"""Convert a local HuggingFace CLAP checkpoint to native clap.cpp GGUF.
Only conversion needs Python/torch/transformers. Input must be trusted: torch
weights_only=True is enforced. Tensor data are streamed from mmap, not duplicated.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import tempfile


def tensor_name(name):
    return (name.replace('audio_model.audio_encoder.', 'audio.')
            .replace('text_model.', 'text.')
            .replace('audio_projection.', 'audio_proj.')
            .replace('text_projection.', 'text_proj.')
            .replace('relative_position_bias_table', 'rel_bias'))


def string(value):
    value = value.encode('utf-8')
    return struct.pack('<Q', len(value)) + value


def convert(source, destination, dtype='f32'):
    import numpy as np
    import torch
    from transformers import ClapConfig
    source, destination = Path(source), Path(destination)
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
    if tokenizer['model']['type'] != 'BPE':
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
    checkpoint = source / 'pytorch_model.bin'
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    tensors = []
    for name, value in weights.items():
        # Integer position/index buffers are reconstructed from config.
        if not value.is_floating_point():
            continue
        name = tensor_name(name)
        if len(name.encode()) >= 64:
            raise ValueError(f'tensor name exceeds ggml limit: {name}')
        if not name.startswith(('audio.', 'text.', 'audio_proj.', 'text_proj.', 'logit_scale')):
            raise ValueError(f'unrecognized tensor: {name}')
        half = dtype == 'f16' and value.ndim > 1
        tensors.append((name, value, np.dtype('<f2' if half else '<f4'), 1 if half else 0))
    metadata = {'general.architecture': 'clap', 'general.name': 'laion/larger_clap_music',
                'clap.config': json.dumps(config, separators=(',', ':')),
                'clap.tokenizer': json.dumps(tokenizer, ensure_ascii=False, separators=(',', ':'))}
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += string(key) + struct.pack('<I', 8) + string(value)
    offset = 0
    for name, value, dt, typ in tensors:
        header += string(name) + struct.pack('<I', value.ndim)
        header += struct.pack('<' + 'Q' * value.ndim, *reversed(value.shape))
        header += struct.pack('<IQ', typ, offset)
        size = value.numel() * dt.itemsize
        offset += (size + 31) // 32 * 32
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(header)
            f.write(b'\0' * ((-len(header)) % 32))
            for name, value, dt, typ in tensors:
                data = value.detach().to(dtype=torch.float16 if typ else torch.float32).contiguous().numpy()
                data.astype(dt, copy=False).tofile(f)
                f.write(b'\0' * ((-data.nbytes) % 32))
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    print(f'{destination}: {len(tensors)} tensors ({dtype})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--dtype', choices=['f32', 'f16'], default='f32')
    args = parser.parse_args()
    convert(args.source, args.destination, args.dtype)
