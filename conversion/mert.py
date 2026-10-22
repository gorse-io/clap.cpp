"""MERT-v1-95M configuration validation and GGUF conversion."""
import json
import struct

from .base import ModelBase, load_checkpoint, write_gguf


def mert_inputs(source, revision, torch):
    """MERT-v1-95M/HuBERT post-norm inference; never execute checkpoint code."""
    if not revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        raise ValueError('MERT conversion requires --mert-revision with a full 40-character commit SHA')
    raw = json.loads((source / 'config.json').read_text())
    p = json.loads((source / 'preprocessor_config.json').read_text())
    for key, expected in dict(feature_extractor_type='Wav2Vec2FeatureExtractor', feature_size=1,
                              sampling_rate=24000, do_normalize=True, padding_value=0,
                              padding_side='right', return_attention_mask=True).items():
        actual = p.get(key)
        if (actual != expected or
            (type(expected) is bool and type(actual) is not bool) or
            (key in ['feature_size', 'sampling_rate'] and type(actual) is not int) or
            (key == 'padding_value' and type(actual) not in [int, float])):
            raise ValueError(f'unsupported MERT preprocessor {key}')
    defaults = dict(feature_extractor_cqt=False, deepnorm=False, do_stable_layer_norm=False,
                    feat_proj_layer_norm=True, conv_pos_batch_norm=False)
    for key, default in defaults.items():
        if type(raw.get(key, default)) is not bool:
            raise ValueError(f'invalid MERT boolean {key}')
    if type(raw.get('conv_bias')) is not bool:
        raise ValueError('invalid MERT boolean conv_bias')
    if (raw.get('model_type') != 'mert_model' or raw.get('sample_rate') != 24000 or
        raw.get('feature_extractor_cqt', False) or raw.get('deepnorm', False) or
        raw.get('do_stable_layer_norm', False) or raw.get('feat_extract_norm') != 'group' or
        not raw.get('feat_proj_layer_norm', True) or raw.get('hidden_act') != 'gelu' or
        raw.get('feat_extract_activation') != 'gelu' or raw.get('conv_pos_batch_norm', False)):
        raise ValueError('unsupported MERT-v1 architecture (expected 95M post-norm group frontend)')
    def positive_int(value):
        return type(value) is int and 0 < value <= 2**31 - 1

    for key in ['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
                'num_conv_pos_embeddings','num_conv_pos_embedding_groups']:
        if not positive_int(raw.get(key)):
            raise ValueError(f'invalid MERT {key}')
    dims, kernels, strides = (raw.get(k, []) for k in ['conv_dim','conv_kernel','conv_stride'])
    if (any(not isinstance(arr, list) for arr in [dims, kernels, strides]) or not dims or
        not len(dims) == len(kernels) == len(strides) or any(
            not positive_int(v) for arr in [dims,kernels,strides] for v in arr)):
        raise ValueError('invalid MERT convolution dimensions')
    if raw['hidden_size'] % raw['num_attention_heads'] or raw['hidden_size'] % raw['num_conv_pos_embedding_groups']:
        raise ValueError('invalid MERT attention/position groups')
    layers = raw.get('num_feat_extract_layers', len(dims))
    if not positive_int(layers) or layers != len(dims):
        raise ValueError('invalid MERT convolution geometry: num_feat_extract_layers')
    receptive, jump = 1, 1
    for kernel, stride in zip(kernels, strides):
        receptive += (kernel - 1) * jump
        jump *= stride
        if receptive > 24000 or jump > 24000:
            raise ValueError('unsupported MERT convolution receptive field')
    import math
    epsilon = raw.get('layer_norm_eps')
    if type(epsilon) not in [int, float]:
        raise ValueError('invalid MERT layer_norm_eps')
    try:
        epsilon = struct.unpack('<f', struct.pack('<f', epsilon))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError('invalid MERT layer_norm_eps') from error
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('invalid MERT layer_norm_eps')
    weights = load_checkpoint(source / 'pytorch_model.bin', torch)
    prefix = 'encoder.pos_conv_embed.conv.'
    # Exactly one direct, legacy or modern representation is required. Never
    # silently drop one half of a pair or overwrite a conflicting dense weight.
    legacy = [prefix+'weight_g', prefix+'weight_v']
    modern = [prefix+'parametrizations.weight.original0', prefix+'parametrizations.weight.original1']
    representations = [prefix+'weight' in weights,
                       any(key in weights for key in legacy), any(key in weights for key in modern)]
    if sum(representations) != 1:
        raise ValueError('missing or conflicting MERT positional weights')
    if not representations[0]:
        pair = legacy if representations[1] else modern
        if not all(key in weights for key in pair):
            raise ValueError('incomplete MERT positional weight-norm pair')
        g, v = (weights.pop(key) for key in pair)
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


@ModelBase.register('mert')
class MertModel(ModelBase):
    def write(self):
        import numpy as np
        import torch
        tensors, metadata = mert_inputs(self.source, self.options.get('mert_revision'), torch)
        write_gguf(self.destination, tensors, metadata, self.dtype, np, torch)
