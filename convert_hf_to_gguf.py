#!/usr/bin/env python3
"""Convert local Hugging Face CLAP, CLaMP3 or MERT2 weights to native GGUF.

Conversion is the only step requiring Python and torch. Checkpoints are loaded
with weights_only=True and mmap=True. CLaMP3 symbolic encoder tensors are not
written because native symbolic input is not supported. MERT2 safetensors are
streamed one tensor at a time so the published 632M-parameter checkpoint can be
converted on memory-constrained hosts.
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


def mert2_inputs(source, revision):
    if not source.is_dir():
        raise ValueError('MERT2 source must be a checkpoint directory')
    if not revision or len(revision) != 40 or any(ch not in '0123456789abcdef' for ch in revision.lower()):
        raise ValueError('MERT2 conversion requires --mert2-revision with a full 40-character commit SHA')
    raw = json.loads((source / 'config.json').read_text())
    if raw.get('model_type') != 'mert2':
        raise ValueError('expected model_type mert2')
    if raw.get('variant') != 'fs':
        raise ValueError("only MERT-v2-FullSong variant 'fs' is supported")
    positive = ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
                'num_mel_bins', 'sampling_rate', 'n_fft', 'win_length', 'hop_length',
                'conv_depthwise_kernel_size', 'rotary_embedding_base')
    if any(not isinstance(raw.get(key), (int, float)) or raw[key] <= 0 for key in positive):
        raise ValueError('invalid MERT2 configuration dimensions')
    if raw['hidden_size'] % raw['num_attention_heads'] or (raw['hidden_size'] // raw['num_attention_heads']) % 2:
        raise ValueError('MERT2 attention head dimension must be even')
    if raw['n_fft'] != raw['win_length'] or raw['n_fft'] & (raw['n_fft'] - 1):
        raise ValueError('native MERT2 requires equal power-of-two n_fft and win_length')
    if raw.get('inputs_to_logits_ratio') != raw['hop_length'] * 4:
        raise ValueError('invalid MERT2 inputs_to_logits_ratio')
    if raw.get('minimum_input_samples') != raw['n_fft'] // 2 + 1:
        raise ValueError('invalid MERT2 minimum_input_samples')
    if raw.get('subsampling_channels', [None])[-1] != raw['hidden_size'] or \
            raw.get('subsampling_channels', [None])[0] != raw['num_mel_bins'] or \
            len(raw.get('subsampling_channels', [])) != 3 or len(raw.get('subsampling_depths', [])) != 3:
        raise ValueError('invalid MERT2 subsampling configuration')
    if raw.get('context_seconds', 0) < 30:
        raise ValueError('MERT-v2-FullSong context metadata must cover at least 30 seconds')
    preprocessing = json.loads((source / 'preprocessor_config.json').read_text())
    required = {'feature_extractor_type': 'Wav2Vec2FeatureExtractor', 'feature_size': 1,
                'sampling_rate': raw['sampling_rate'], 'padding_value': 0.0,
                'padding_side': 'right', 'return_attention_mask': True, 'do_normalize': False}
    for key, expected in required.items():
        if preprocessing.get(key) != expected:
            raise ValueError(f'unsupported MERT2 preprocessor {key}: {preprocessing.get(key)}')
    checkpoint = source / 'model.safetensors'
    if not checkpoint.is_file():
        raise ValueError('MERT2 source is missing model.safetensors')
    metadata = {'general.architecture': 'mert2', 'general.name': 'm-a-p/MERT-v2-FullSong',
                'mert2.config': json.dumps(raw, separators=(',', ':')),
                'mert2.source_revision': revision.lower()}
    return checkpoint, metadata


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


def write_safetensors_gguf(destination, checkpoint, metadata, dtype, np, torch):
    from safetensors import safe_open
    with safe_open(checkpoint, framework='pt', device='cpu') as source:
        encoded = []
        for name in source.keys():
            if len(name.encode()) >= 64:
                raise ValueError(f'tensor name exceeds ggml limit: {name}')
            if not name.startswith(('feature_extractor.', 'subsampling_module.', 'layers.')):
                raise ValueError(f'unrecognized MERT2 tensor: {name}')
            view = source.get_slice(name)
            if view.get_dtype() != 'F32':
                raise ValueError(f'MERT2 source tensor must be F32: {name}')
            shape = tuple(view.get_shape())
            half = dtype == 'f16' and len(shape) > 1 and not name.startswith('feature_extractor.')
            item_size = 2 if half else 4
            elements = 1
            for dimension in shape:
                elements *= dimension
            encoded.append((name, shape, np.dtype('<f2' if half else '<f4'), 1 if half else 0,
                            elements, item_size))
        header = b'GGUF' + struct.pack('<IQQ', 3, len(encoded), len(metadata))
        for key, value in metadata.items():
            header += string(key) + struct.pack('<I', 8) + string(value)
        offset = 0
        for name, shape, _, typ, elements, item_size in encoded:
            header += string(name) + struct.pack('<I', len(shape))
            header += struct.pack('<' + 'Q' * len(shape), *reversed(shape))
            header += struct.pack('<IQ', typ, offset)
            offset += (elements * item_size + 31) // 32 * 32
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as output:
                output.write(header)
                output.write(b'\0' * ((-len(header)) % 32))
                for name, _, dt, typ, _, _ in encoded:
                    value = source.get_tensor(name)
                    data = value.detach().to(dtype=torch.float16 if typ else torch.float32).contiguous().numpy()
                    data.astype(dt, copy=False).tofile(output)
                    output.write(b'\0' * ((-data.nbytes) % 32))
                    del data, value
            os.replace(temp, destination)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    print(f'{destination}: {len(encoded)} tensors ({dtype}, mert2)')


def mert_inputs(source, revision, torch):
    """MERT-v1-95M/HuBERT post-norm inference; never execute checkpoint code."""
    if not revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        raise ValueError('MERT conversion requires --mert-revision with a full 40-character commit SHA')
    raw = json.loads((source / 'config.json').read_text())
    p = json.loads((source / 'preprocessor_config.json').read_text())
    for key, expected in dict(feature_extractor_type='Wav2Vec2FeatureExtractor', feature_size=1,
                              sampling_rate=24000, do_normalize=True, padding_value=0,
                              padding_side='right', return_attention_mask=True).items():
        if p.get(key) != expected:
            raise ValueError(f'unsupported MERT preprocessor {key}')
    if (raw.get('model_type') != 'mert_model' or raw.get('sample_rate') != 24000 or
        raw.get('feature_extractor_cqt', False) or raw.get('deepnorm', False) or
        raw.get('do_stable_layer_norm', False) or raw.get('feat_extract_norm') != 'group' or
        not raw.get('feat_proj_layer_norm', True) or raw.get('hidden_act') != 'gelu' or
        raw.get('feat_extract_activation') != 'gelu' or raw.get('conv_pos_batch_norm', False)):
        raise ValueError('unsupported MERT-v1 architecture (expected 95M post-norm group frontend)')
    for key in ['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
                'num_conv_pos_embeddings','num_conv_pos_embedding_groups']:
        if type(raw.get(key)) is not int or raw[key] <= 0:
            raise ValueError(f'invalid MERT {key}')
    dims, kernels, strides = (raw.get(k, []) for k in ['conv_dim','conv_kernel','conv_stride'])
    if not dims or not len(dims) == len(kernels) == len(strides) or any(
            type(v) is not int or v <= 0 for arr in [dims,kernels,strides] for v in arr):
        raise ValueError('invalid MERT convolution dimensions')
    if raw['hidden_size'] % raw['num_attention_heads'] or raw['hidden_size'] % raw['num_conv_pos_embedding_groups']:
        raise ValueError('invalid MERT attention/position groups')
    if raw.get('num_feat_extract_layers', len(dims)) != len(dims):
        raise ValueError('invalid MERT convolution geometry: num_feat_extract_layers')
    import math
    if not math.isfinite(raw.get('layer_norm_eps', 0)) or raw['layer_norm_eps'] <= 0:
        raise ValueError('invalid MERT layer_norm_eps')
    weights = load_checkpoint(source / 'pytorch_model.bin', torch)
    prefix = 'encoder.pos_conv_embed.conv.'
    # Legacy and modern PyTorch weight_norm share the same dim=2 contract.
    g = weights.pop(prefix+'weight_g', None)
    v = weights.pop(prefix+'weight_v', None)
    if g is None:
        g = weights.pop(prefix+'parametrizations.weight.original0', None)
        v = weights.pop(prefix+'parametrizations.weight.original1', None)
    if g is not None and v is not None:
        weights[prefix+'weight'] = torch._weight_norm(v, g, 2)
    selected=[]
    for name,value in weights.items():
        if name == 'masked_spec_embed': continue  # training-only masking
        if not name.startswith(('feature_extractor.', 'feature_projection.', 'encoder.')):
            raise ValueError(f'unrecognized MERT tensor: {name}')
        selected.append((name.replace('feature_extractor.conv_layers.', 'frontend.')
                         .replace('feature_projection.', 'projection.')
                         .replace('encoder.pos_conv_embed.conv.', 'position.')
                         .replace('encoder.layers.', 'layers.')
                         .replace('.feed_forward.intermediate_dense.', '.ffn.in.')
                         .replace('.feed_forward.output_dense.', '.ffn.out.'),value))
    return selected, {'general.architecture':'mert', 'general.name':'m-a-p/MERT-v1-95M',
                      'mert.config':json.dumps(raw,separators=(',',':')),
                      'mert.source_revision':revision.lower()}


def convert(source, destination, dtype='f32', clamp3_config=None, tokenizer=None, mert2_revision=None,
            mert_revision=None):
    import numpy as np
    import torch
    source, destination = Path(source), Path(destination)
    raw_config = json.loads((source / 'config.json').read_text()) if source.is_dir() and (source / 'config.json').exists() else {}
    if raw_config.get('model_type') == 'mert2':
        checkpoint, metadata = mert2_inputs(source, mert2_revision)
        write_safetensors_gguf(destination, checkpoint, metadata, dtype, np, torch)
        return
    if raw_config.get('model_type') == 'mert_model':
        tensors, metadata = mert_inputs(source, mert_revision, torch)
    elif clamp3_config is not None or source.is_file():
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
    parser.add_argument('--mert2-revision', help='full pinned Hugging Face revision SHA for MERT2 metadata')
    parser.add_argument('--mert-revision', help='full pinned Hugging Face revision SHA for MERT-v1 metadata')
    args = parser.parse_args()
    convert(args.source, args.destination, args.dtype, args.clamp3_config, args.tokenizer, args.mert2_revision,
            args.mert_revision)
