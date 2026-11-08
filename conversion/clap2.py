"""CLAP2 HF-name GGUF conversion, bounded-memory safetensors streaming.

Reference: mlfoundations/open_clip at OPEN_CLIP_REVISION (audio/naflex_audio.py,
audio/naflex_tower.py and transformer.ModernTextTransformer). The Transformers
export uses different names from open_clip_model.safetensors: only the HF
model.safetensors checkpoint is accepted, without renaming or transposition.
"""
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

from .base import ModelBase, string

# Published tower dimensions (width, FFN width, heads, depth), projection and
# immutable HF revisions. In particular XXS uses 56-dimensional attention heads.
PUBLISHED_MODELS = {
    'large': ((1024, 2816, 16, 12), (960, 2560, 15, 11), 960,
              'e836feaa79cd1f7de939fe77065819f48fc44e4c'),
    'medium': ((640, 1728, 10, 12), (640, 1728, 10, 11), 640,
               '3a8d9d79cb98e4d409fce24092a69718a821a7ad'),
    'small': ((512, 1408, 8, 12), (512, 1408, 8, 11), 512,
              '9891ee368dbec866dab1919f163b7e091ab0ed37'),
    'xs': ((320, 896, 5, 12), (320, 896, 5, 12), 320,
           'adc510748f0b759918ba4aaee94fc864e05c48d9'),
    'xxs': ((224, 640, 4, 12), (224, 640, 4, 12), 224,
            'adaca9be91f5045d06e10569de3075e1ef37f823'),
}
OPEN_CLIP_REVISION = 'f9a0948d573009f8fa3e530e55814cb11fb8e8bd'


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


def _expect(config, expected, label):
    for key, value in expected.items():
        actual = config.get(key)
        if actual != value or (isinstance(value, bool) and type(actual) is not bool):
            raise ValueError(f'unsupported CLAP2 {label}.{key}: {actual!r}; expected {value!r}')


def _positive(config, names, label):
    for key in names:
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f'invalid CLAP2 {label}.{key}')


def validate_config(config, processor):
    """Validate the supported axial-audio / bidirectional MAP-text computation."""
    _expect(config, {'model_type': 'clap2', 'projection_hidden_act': 'gelu'}, 'config')
    _positive(config, ('projection_dim',), 'config')
    for branch in ('audio', 'text'):
        c = config.get(branch + '_config')
        if not isinstance(c, dict):
            raise ValueError(f'missing CLAP2 {branch}_config')
        _positive(c, ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_hidden_layers'), branch)
        _expect(c, {'model_type': f'clap2_{branch}_model', 'hidden_act': 'silu',
                    'attention_dropout': 0.0, 'attention_bias': branch == 'audio',
                    'mlp_bias': branch == 'audio', 'use_qk_norm': branch == 'text'}, branch)
        width, heads = c['hidden_size'], c['num_attention_heads']
        divisor = 4 if branch == 'audio' else 2
        if width % heads or (width // heads) % divisor:
            raise ValueError(f'invalid CLAP2 {branch} RoPE head dimension')
        eps = c.get('rms_norm_eps')
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError(f'invalid CLAP2 {branch}.rms_norm_eps')
        rope = c.get('rope_parameters')
        if not isinstance(rope, dict) or set(rope) != {'rope_type', 'rope_theta'}:
            raise ValueError(f'unsupported CLAP2 {branch}.rope_parameters')
        _expect(rope, {'rope_type': 'axial' if branch == 'audio' else 'default'}, branch + '.rope_parameters')
        theta = rope.get('rope_theta')
        if type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0:
            raise ValueError(f'invalid CLAP2 {branch}.rope_theta')
        # Reject computation-changing extensions rather than silently treating
        # a future configuration as the currently supported architecture.
        for key, default in {'attention_mode': 'bidirectional', 'pool_type': 'map',
                             'norm_type': 'rmsnorm', 'norm_placement': 'pre',
                             'attn_gated': False, 'value_residual': False}.items():
            if key in c:
                _expect(c, {key: default}, branch)
    audio, text = config['audio_config'], config['text_config']
    _positive(audio, ('num_mel_bins', 'patch_size'), 'audio')
    _expect(audio, {'num_register_tokens': 1, 'pooler_hidden_act': 'gelu'}, 'audio')
    scale = audio.get('layerscale_value')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('invalid CLAP2 audio.layerscale_value')
    if audio['num_mel_bins'] % audio['patch_size']:
        raise ValueError('CLAP2 mel bins must be divisible by patch_size')
    _positive(text, ('max_position_embeddings', 'vocab_size'), 'text')
    _expect(text, {'layerscale_value': None}, 'text')
    ids = [text.get(k + '_token_id') for k in ('bos', 'eos', 'pad')]
    if any(type(i) is not int or not 0 <= i < text['vocab_size'] for i in ids) or len(set(ids)) != 3:
        raise ValueError('invalid CLAP2 special token ids')
    _expect(processor, {'processor_class': 'Clap2Processor'}, 'processor')
    feature = processor.get('feature_extractor')
    if not isinstance(feature, dict):
        raise ValueError('missing CLAP2 feature_extractor')
    _expect(feature, {'feature_extractor_type': 'Clap2FeatureExtractor',
                     'feature_size': audio['num_mel_bins'], 'patch_size': audio['patch_size'],
                     'sampling_rate': 32000, 'n_fft': 1024, 'hop_length': 320,
                     'f_min': 50.0, 'f_max': 16000.0, 'max_num_patches': None,
                     'padding_side': 'right', 'padding_value': 0.0,
                     'return_attention_mask': True}, 'processor.feature_extractor')


def validate_tokenizer(tokenizer, config):
    """Require the exported GPT2 byte-BPE + BOS/EOS template, not CLAP's RoBERTa."""
    model = tokenizer.get('model', {})
    _expect(model, {'type': 'BPE', 'dropout': None, 'unk_token': None,
                    'continuing_subword_prefix': '', 'end_of_word_suffix': '',
                    'fuse_unk': False, 'byte_fallback': False, 'ignore_merges': False}, 'tokenizer.model')
    _expect(tokenizer, {'normalizer': None}, 'tokenizer')
    _expect(tokenizer.get('pre_tokenizer', {}),
            {'type': 'ByteLevel', 'add_prefix_space': False, 'use_regex': True}, 'tokenizer.pre_tokenizer')
    _expect(tokenizer.get('decoder', {}), {'type': 'ByteLevel'}, 'tokenizer.decoder')
    vocab = model.get('vocab')
    merges = model.get('merges')
    if not isinstance(vocab, dict) or not vocab or not isinstance(merges, list):
        raise ValueError('invalid CLAP2 BPE vocabulary or merges')
    text = config['text_config']
    ids = list(vocab.values())
    added = tokenizer.get('added_tokens', [])
    if not isinstance(added, list):
        raise ValueError('invalid CLAP2 added tokens')
    token_ids = dict(vocab)
    for token in added:
        if not isinstance(token, dict) or type(token.get('id')) is not int or not isinstance(token.get('content'), str):
            raise ValueError('invalid CLAP2 added token')
        token_ids[token['content']] = token['id']
    if any(type(i) is not int or not 0 <= i < text['vocab_size'] for i in ids + list(token_ids.values())):
        raise ValueError('CLAP2 tokenizer ids outside vocabulary')
    if len(set(token_ids.values())) != text['vocab_size'] or len(token_ids) != text['vocab_size']:
        raise ValueError('CLAP2 tokenizer/config vocabulary mismatch')
    for key in ('bos', 'eos', 'pad'):
        if token_ids.get(f'<|{key}|>') != text[key + '_token_id']:
            raise ValueError(f'CLAP2 tokenizer {key} token mismatch')
    post = tokenizer.get('post_processor', {})
    _expect(post, {'type': 'TemplateProcessing', 'single': [
        {'SpecialToken': {'id': '<|bos|>', 'type_id': 0}},
        {'Sequence': {'id': 'A', 'type_id': 0}},
        {'SpecialToken': {'id': '<|eos|>', 'type_id': 0}}]}, 'tokenizer.post_processor')
    for key in ('bos', 'eos'):
        special = post.get('special_tokens', {}).get(f'<|{key}|>', {})
        _expect(special, {'ids': [text[key + '_token_id']], 'tokens': [f'<|{key}|>']}, 'tokenizer.special_tokens')


def expected_tensor_shapes(config):
    """Complete HF schema; validates both missing and unsupported parameters."""
    a, t = config['audio_config'], config['text_config']
    aw, tw, p = a['hidden_size'], t['hidden_size'], config['projection_dim']
    tensors = {'logit_scale': (),
               'audio_model.embeddings.patch_embedding.weight': (aw, a['patch_size'] ** 2),
               'audio_model.embeddings.register_tokens': (1, 1, aw),
               'audio_model.embeddings.layernorm.weight': (aw,),
               'audio_model.norm.weight': (aw,),
               'text_model.embed_tokens.weight': (t['vocab_size'], tw),
               'text_model.norm.weight': (tw,),
               'text_projection.weight': (p, tw)}

    def linear(prefix, out_dim, in_dim, bias):
        tensors[prefix + '.weight'] = (out_dim, in_dim)
        if bias:
            tensors[prefix + '.bias'] = (out_dim,)

    for branch, c in (('audio', a), ('text', t)):
        width, hidden = c['hidden_size'], c['intermediate_size']
        for layer in range(c['num_hidden_layers']):
            prefix = f'{branch}_model.encoder.layers.{layer}.'
            for norm in ('input_layernorm', 'post_attention_layernorm'):
                tensors[prefix + norm + '.weight'] = (width,)
            linear(prefix + 'self_attn.qkv_proj', 3 * width, width, c['attention_bias'])
            linear(prefix + 'self_attn.o_proj', width, width, c['attention_bias'])
            linear(prefix + 'mlp.gate_up_proj', 2 * hidden, width, c['mlp_bias'])
            linear(prefix + 'mlp.down_proj', width, hidden, c['mlp_bias'])
            if branch == 'audio':
                for number in (1, 2):
                    tensors[prefix + f'layer_scale{number}.lambda1'] = (width,)
            else:
                for norm in ('q_norm', 'k_norm'):
                    tensors[prefix + 'self_attn.' + norm + '.weight'] = (width // c['num_attention_heads'],)
        prefix = branch + '_model.head.'
        tensors[prefix + 'probe'] = (1, 1, width)
        linear(prefix + 'q_proj', width, width, c['attention_bias'])
        linear(prefix + 'kv_proj', 2 * width, width, c['attention_bias'])
        if branch == 'audio':
            linear(prefix + 'o_proj', width, width, True)
            tensors[prefix + 'layernorm.weight'] = (width,)
            linear(prefix + 'fc1', hidden, width, True)
            linear(prefix + 'fc2', width, hidden, True)
        else:
            for norm in ('q_norm', 'k_norm'):
                tensors[prefix + norm + '.weight'] = (width // c['num_attention_heads'],)
    linear('audio_projection.linear1', p, aw, True)
    linear('audio_projection.linear2', p, p, True)
    return tensors


def clap2_inputs(source, revision=None, tokenizer_path=None):
    source = Path(source)
    if not source.is_dir():
        raise ValueError('CLAP2 source must be a checkpoint directory')
    if revision is not None and (not isinstance(revision, str) or re.fullmatch(r'[0-9a-fA-F]{40}', revision) is None):
        raise ValueError('CLAP2 revision must be a full 40-character commit SHA')
    config = _read_json(source / 'config.json')
    processor = _read_json(source / 'processor_config.json')
    tokenizer = _read_json(tokenizer_path or source / 'tokenizer.json')
    validate_config(config, processor)
    validate_tokenizer(tokenizer, config)
    checkpoint = source / 'model.safetensors'
    if not checkpoint.is_file():
        raise ValueError('CLAP2 source is missing HF model.safetensors (open_clip weights are not interchangeable)')
    metadata = {'general.architecture': 'clap2', 'general.name': 'CLAP2',
                'clap2.open_clip_revision': OPEN_CLIP_REVISION,
                'clap2.config': json.dumps(config, separators=(',', ':')),
                'clap2.processor_config': json.dumps(processor, separators=(',', ':')),
                'clap2.tokenizer': json.dumps(tokenizer, separators=(',', ':'), ensure_ascii=False)}
    fields = ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_hidden_layers')
    signature = tuple(tuple(config[branch + '_config'][key] for key in fields)
                      for branch in ('audio', 'text')) + (config['projection_dim'],)
    for size, (*dimensions, pinned_revision) in PUBLISHED_MODELS.items():
        if signature == tuple(dimensions):
            repo = f'laion/clap2-{size}-audioxl-s1B'
            metadata.update({'general.name': repo, 'clap2.source_repo': repo})
            revision = pinned_revision if revision is None else revision
            break
    if revision is not None:
        metadata['clap2.source_revision'] = revision.lower()
    return checkpoint, metadata, config


def _safetensors_header(source):
    raw = source.read(8)
    if len(raw) != 8:
        raise ValueError('truncated CLAP2 safetensors header')
    size, = struct.unpack('<Q', raw)
    if not 2 <= size <= 16 * 1024 * 1024:
        raise ValueError('invalid CLAP2 safetensors header size')
    raw = source.read(size)
    if len(raw) != size:
        raise ValueError('truncated CLAP2 safetensors header')

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate safetensors key: {key}')
            result[key] = value
        return result

    header = json.loads(raw, object_pairs_hook=unique_keys)
    if not isinstance(header, dict):
        raise ValueError('invalid CLAP2 safetensors header')
    header.pop('__metadata__', None)
    return header, 8 + size


def write_clap2_gguf(destination, checkpoint, metadata, config, dtype):
    """Copy F32 or cast F16 in 4 MiB chunks, never load a checkpoint/tensor."""
    if dtype not in ('f32', 'f16'):
        raise ValueError(f'unsupported CLAP2 output dtype: {dtype}')
    import numpy as np
    destination, checkpoint = Path(destination), Path(checkpoint)
    with checkpoint.open('rb') as source:
        tensors, data_start = _safetensors_header(source)
        expected = expected_tensor_shapes(config)
        if set(tensors) != set(expected):
            raise ValueError(f'CLAP2 tensor schema mismatch: missing={sorted(set(expected) - set(tensors))}; '
                             f'unexpected={sorted(set(tensors) - set(expected))}')
        encoded, ranges = [], []
        for name in sorted(tensors):
            tensor = tensors[name]
            shape = tensor.get('shape')
            if not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape) or tuple(shape) != expected[name]:
                raise ValueError(f'CLAP2 tensor shape mismatch: {name}: {shape!r}, expected {expected[name]}')
            if tensor.get('dtype') != 'F32':
                raise ValueError(f'CLAP2 source tensor must be F32: {name}')
            if len(name.encode('utf-8')) >= 64:
                raise ValueError(f'tensor name exceeds ggml limit: {name}')
            offsets = tensor.get('data_offsets')
            count = math.prod(shape)
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets) or offsets[1] - offsets[0] != 4 * count:
                raise ValueError(f'invalid CLAP2 safetensors offsets: {name}')
            ranges.append(tuple(offsets))
            # Learned register/probe vectors remain F32, like norms and biases.
            half = dtype == 'f16' and len(shape) == 2
            encoded.append((name, shape or [1], half, offsets[0], count))
        end = 0
        for start, stop in sorted(ranges):
            if start != end:
                raise ValueError('CLAP2 safetensors offsets overlap or contain gaps')
            end = stop
        if data_start + end != os.fstat(source.fileno()).st_size:
            raise ValueError('CLAP2 safetensors file size does not match header')
        header = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(encoded), len(metadata)))
        for key, value in metadata.items():
            header += string(key) + struct.pack('<I', 8) + string(value)
        offset = 0
        for name, shape, half, _, count in encoded:
            header += string(name) + struct.pack('<I', len(shape))
            header += struct.pack('<' + 'Q' * len(shape), *reversed(shape))
            header += struct.pack('<IQ', int(half), offset)
            offset += (count * (2 if half else 4) + 31) // 32 * 32
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as output:
                output.write(header)
                output.write(b'\0' * ((-len(header)) % 32))
                for name, _, half, start, count in encoded:
                    source.seek(data_start + start)
                    remaining = count * 4
                    while remaining:
                        chunk = source.read(min(remaining, 4 * 1024 * 1024))
                        if not chunk or len(chunk) % 4:
                            raise ValueError(f'truncated CLAP2 tensor: {name}')
                        if half:
                            output.write(np.frombuffer(chunk, dtype='<f4').astype('<f2').tobytes())
                        else:
                            output.write(chunk)
                        remaining -= len(chunk)
                    output.write(b'\0' * ((-count * (2 if half else 4)) % 32))
            os.replace(temp, destination)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    print(f'{destination}: {len(encoded)} tensors ({dtype}, clap2)')


@ModelBase.register('clap2')
class Clap2Model(ModelBase):
    def write(self):
        checkpoint, metadata, config = clap2_inputs(self.source, self.options.get('clap2_revision'),
                                                    self.options.get('tokenizer'))
        write_clap2_gguf(self.destination, checkpoint, metadata, config, self.dtype)
