"""MuQ configuration validation and streaming safetensors conversion."""
import json
import math
import os
import struct
import tempfile

from .base import ModelBase, string


def muq_inputs(source, revision):
    if not revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('MuQ conversion requires --muq-revision with a full lowercase commit SHA')
    raw = json.loads((source / 'config.json').read_text())
    if raw.get('features') != ['melspec_2048'] or raw.get('n_mels') != 128 or raw.get('label_rate') != 25:
        raise ValueError('only MuQ 128-bin melspec_2048, 25 Hz output is supported')
    if raw.get('is_flash', False):
        raise ValueError('MuQ flash variant is unsupported')
    if raw.get('hop_length') != 240:
        raise ValueError('native MuQ requires hop_length 240')
    for key in ('encoder_dim', 'encoder_depth', 'conv_dim', 'hop_length'):
        if type(raw.get(key)) is not int or raw[key] <= 0:
            raise ValueError('invalid MuQ dimension: ' + key)
    cfg = raw['w2v2_config']
    heads = cfg.get('num_attention_heads', 0)
    if type(heads) is not int or heads <= 0 or raw['encoder_dim'] % heads or (raw['encoder_dim'] // heads) % 2:
        raise ValueError('invalid MuQ attention head geometry')
    if cfg.get('position_embeddings_type') != 'rotary' or cfg.get('hidden_act') not in ('swish', 'silu'):
        raise ValueError('MuQ requires rotary embeddings and swish activation')
    for key in ('intermediate_size', 'conv_depthwise_kernel_size'):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ValueError('invalid MuQ ' + key)
    if cfg['conv_depthwise_kernel_size'] % 2 != 1:
        raise ValueError('MuQ depthwise kernel must be odd')
    for value in (cfg.get('layer_norm_eps'), cfg.get('rotary_embedding_base'), raw['stat'].get('melspec_2048_std')):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid MuQ normalization/rotary configuration')
    if not math.isfinite(raw['stat']['melspec_2048_mean']):
        raise ValueError('invalid MuQ feature mean')
    metadata = {'general.architecture': 'muq', 'general.name': 'OpenMuQ/MuQ-large-msd-iter',
                'general.license': 'cc-by-nc-4.0', 'muq.source_revision': revision,
                'muq.config': json.dumps(raw, separators=(',', ':'))}
    return source / 'model.safetensors', metadata


def write_muq_gguf(destination, checkpoint, metadata, dtype, np, torch):
    from safetensors import safe_open
    import torchaudio
    # The official frontend is not an nn.Module and is absent from state_dict.
    frontend = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=2048,
                                                   hop_length=json.loads(metadata['muq.config'])['hop_length'], n_mels=128)
    extra = {'frontend.window': frontend.spectrogram.window, 'frontend.filters': frontend.mel_scale.fb}
    with safe_open(checkpoint, framework='pt', device='cpu') as source:
        encoded = []
        for original in source.keys():
            if original.endswith('.num_batches_tracked'):
                continue
            if original.startswith(('model.conv.', 'model.conformer.layers.', 'model.conformer.layer_norm.')):
                name = original.removeprefix('model.')
                view = source.get_slice(original)
                if view.get_dtype() != 'F32':
                    raise ValueError('MuQ source weights must be F32: ' + original)
                shape = tuple(view.get_shape())
            elif original.startswith(('model.rvq.', 'model.linear.', 'model.conformer.pos_conv_embed.')) or original == 'model.cls_token':
                continue  # Training-only or unused by the official inference forward.
            elif original == 'model.conformer.embed_positions.inv_freq':
                raw = json.loads(metadata['muq.config'])
                head = raw['encoder_dim'] // raw['w2v2_config']['num_attention_heads']
                expected = 1.0 / (raw['w2v2_config']['rotary_embedding_base'] ** (torch.arange(0, head, 2).float() / head))
                if not torch.allclose(source.get_tensor(original), expected, rtol=1e-6, atol=0):
                    raise ValueError('MuQ checkpoint rotary frequencies disagree with config')
                continue
            else:
                raise ValueError('unrecognized MuQ tensor: ' + original)
            encoded.append((name, original, shape))
        encoded += [(name, None, tuple(value.shape)) for name, value in extra.items()]
        if not any(name.startswith('conformer.layers.') for name, _, _ in encoded):
            raise ValueError('MuQ checkpoint has no encoder tensors')
        header = b'GGUF' + struct.pack('<IQQ', 3, len(encoded), len(metadata))
        for key, value in metadata.items():
            header += string(key) + struct.pack('<I', 8) + string(value)
        offset = 0
        for name, original, shape in encoded:
            if len(name.encode()) >= 64:
                raise ValueError('tensor name exceeds ggml limit: ' + name)
            half = dtype == 'f16' and original is not None and len(shape) > 1
            header += string(name) + struct.pack('<I', len(shape))
            header += struct.pack('<' + 'Q' * len(shape), *reversed(shape)) + struct.pack('<IQ', int(half), offset)
            elements = math.prod(shape)
            offset += (elements * (2 if half else 4) + 31) // 32 * 32
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as output:
                output.write(header + b'\0' * (-len(header) % 32))
                for name, original, shape in encoded:
                    value = source.get_tensor(original) if original else extra[name]
                    half = dtype == 'f16' and original is not None and len(shape) > 1
                    data = value.to(dtype=torch.float16 if half else torch.float32).contiguous().numpy()
                    data.astype(np.dtype('<f2' if half else '<f4'), copy=False).tofile(output)
                    output.write(b'\0' * (-data.nbytes % 32))
                    del data, value
            os.replace(temp, destination)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    print(f'{destination}: {len(encoded)} tensors ({dtype}, muq)')


@ModelBase.register('muq')
class MuQModel(ModelBase):
    def write(self):
        import numpy as np
        import torch
        checkpoint, metadata = muq_inputs(self.source, self.options.get('muq_revision'))
        write_muq_gguf(self.destination, checkpoint, metadata, self.dtype, np, torch)
