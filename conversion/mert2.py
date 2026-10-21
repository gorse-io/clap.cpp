"""MERT2 configuration validation and streaming safetensors conversion."""
import json
import os
import struct
import tempfile

from .base import ModelBase, string


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


@ModelBase.register('mert2')
class Mert2Model(ModelBase):
    def write(self):
        import numpy as np
        import torch
        checkpoint, metadata = mert2_inputs(self.source, self.options.get('mert2_revision'))
        write_safetensors_gguf(self.destination, checkpoint, metadata, self.dtype, np, torch)
