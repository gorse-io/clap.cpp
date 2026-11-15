# Copyright 2026 clap.cpp Authors
# SPDX-License-Identifier: Apache-2.0
"""VoiceCLAP merged Qwen2.5-Omni audio/text checkpoint conversion.

Never imports checkpoint Python code or materializes the checkpoint in RAM.
"""
import hashlib
import json
import math
from pathlib import Path
import struct

from .base import ModelBase, write_gguf_streaming

VOICECLAP_REVISION = 'ae7e1d0430d131ff1c07fdaecb813637f4c0c8b0'
_TEMPLATE_SHA256 = '3c1d5ac14fd922a04795e7484dc7038275a2fd545f94caf6123eb2b34708ae92'


def tensor_name(name):
    if name == 'lm_head.weight' or name.startswith('visual.'):
        return None
    if name.startswith('audio_tower.'):
        name = 'audio.' + name[len('audio_tower.'):]
    elif name.startswith('model.'):
        name = 'text.' + name[len('model.'):]
    else:
        raise ValueError(f'unrecognized VoiceCLAP tensor: {name}')
    name = (name.replace('.self_attn.', '.attn.')
            .replace('.input_layernorm.', '.attn_norm.')
            .replace('.post_attention_layernorm.', '.ffn_norm.')
            .replace('.self_attn_layer_norm.', '.attn_norm.')
            .replace('.final_layer_norm.', '.ffn_norm.'))
    if len(name.encode('utf-8')) >= 64:
        raise ValueError(f'tensor name exceeds ggml limit: {name}')
    return name


def _json(source, name):
    try:
        return json.loads((source / name).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'missing or invalid VoiceCLAP {name}') from error


def _require(actual, expected, context):
    if not isinstance(actual, dict):
        raise ValueError(f'invalid VoiceCLAP {context}')
    for key, value in expected.items():
        current = actual.get(key)
        if current != value or (type(value) in (bool, int) and type(current) is not type(value)):
            raise ValueError(f'unsupported VoiceCLAP {context}: {key}')


def _validate_config(config):
    _require(config, {'model_type': 'qwen2_5_omni_thinker',
                      'architectures': ['Qwen2_5OmniThinkerForConditionalGeneration']}, 'config')
    text, audio = config.get('text_config', {}), config.get('audio_config', {})
    _require(text, {'model_type': 'qwen2_5_omni_text', 'hidden_act': 'silu',
                    'use_sliding_window': False}, 'text architecture')
    _require(audio, {'model_type': 'qwen2_5_omni_audio_encoder', 'activation_function': 'gelu',
                     'num_mel_bins': 128, 'n_window': 100, 'scale_embedding': False}, 'audio architecture')
    for obj, keys in [(text, ['hidden_size', 'intermediate_size', 'num_hidden_layers',
                             'num_attention_heads', 'num_key_value_heads', 'vocab_size',
                             'max_position_embeddings']),
                      (audio, ['d_model', 'encoder_ffn_dim', 'encoder_layers',
                               'encoder_attention_heads', 'output_dim', 'max_source_positions'])]:
        for key in keys:
            if type(obj.get(key)) is not int or not 0 < obj[key] <= 2**31 - 1:
                raise ValueError(f'invalid VoiceCLAP dimension: {key}')
    h, q, kv = (text[k] for k in ['hidden_size', 'num_attention_heads', 'num_key_value_heads'])
    if (h % q or q % kv or (h // q) % 2 or
            audio['d_model'] % audio['encoder_attention_heads'] or audio['d_model'] % 2 or
            audio['output_dim'] != h or audio.get('num_hidden_layers', audio['encoder_layers']) != audio['encoder_layers']):
        raise ValueError('unsupported VoiceCLAP attention geometry')
    for key in ['rms_norm_eps', 'rope_theta']:
        value = text.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid VoiceCLAP {key}')
    rope = text.get('rope_scaling', {})
    if (rope.get('rope_type', rope.get('type')) != 'default' or
            text.get('layer_types', ['full_attention'] * text['num_hidden_layers']) !=
            ['full_attention'] * text['num_hidden_layers']):
        raise ValueError('unsupported VoiceCLAP rotary/attention variant')
    return text, audio


def _expected_shapes(text, audio):
    h, f, q, kv = (text[k] for k in ['hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads'])
    a, af = audio['d_model'], audio['encoder_ffn_dim']
    shapes = {'model.embed_tokens.weight': [text['vocab_size'], h], 'model.norm.weight': [h],
              'audio_tower.conv1.weight': [a, 128, 3], 'audio_tower.conv1.bias': [a],
              'audio_tower.conv2.weight': [a, a, 3], 'audio_tower.conv2.bias': [a],
              'audio_tower.ln_post.weight': [a], 'audio_tower.ln_post.bias': [a],
              'audio_tower.proj.weight': [h, a], 'audio_tower.proj.bias': [h]}
    for i in range(text['num_hidden_layers']):
        base = f'model.layers.{i}.'
        for key in ['input_layernorm', 'post_attention_layernorm']:
            shapes[base + key + '.weight'] = [h]
        for key, width in [('q', h), ('k', h // q * kv), ('v', h // q * kv), ('o', h)]:
            prefix = base + 'self_attn.' + key + '_proj'
            shapes[prefix + '.weight'] = [width, h]
            if key != 'o':
                shapes[prefix + '.bias'] = [width]
        for key, shape in [('gate', [f, h]), ('up', [f, h]), ('down', [h, f])]:
            shapes[base + 'mlp.' + key + '_proj.weight'] = shape
    for i in range(audio['encoder_layers']):
        base = f'audio_tower.layers.{i}.'
        for key in ['self_attn_layer_norm', 'final_layer_norm']:
            for kind in ['weight', 'bias']:
                shapes[base + key + '.' + kind] = [a]
        for key in ['q', 'k', 'v', 'out']:
            prefix = base + 'self_attn.' + key + '_proj'
            shapes[prefix + '.weight'] = [a, a]
            if key != 'k':
                shapes[prefix + '.bias'] = [a]
        for key, shape in [('fc1', [af, a]), ('fc2', [a, af])]:
            shapes[base + key + '.weight'] = shape
            shapes[base + key + '.bias'] = [shape[0]]
    return shapes


def _safetensors(path):
    """Parse and validate only the safetensors header, returning file slices."""
    try:
        size = path.stat().st_size
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('truncated safetensors prefix')
            length, = struct.unpack('<Q', prefix)
            if not 2 <= length <= min(16 * 1024 * 1024, size - 8):
                raise ValueError('invalid safetensors header length')
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f'duplicate safetensors key: {key}')
                    result[key] = value
                return result
            header = json.loads(stream.read(length), object_pairs_hook=unique)
        if not isinstance(header, dict):
            raise ValueError('invalid safetensors header')
        tensors, intervals = {}, []
        for name, value in header.items():
            if name == '__metadata__':
                continue
            shape, dtype, offsets = value.get('shape'), value.get('dtype'), value.get('data_offsets')
            width = {'BF16': 2, 'F16': 2, 'F32': 4}.get(dtype)
            if (not width or not isinstance(shape, list) or not 1 <= len(shape) <= 4 or
                    any(type(n) is not int or n <= 0 for n in shape) or
                    not isinstance(offsets, list) or len(offsets) != 2 or
                    any(type(n) is not int or n < 0 for n in offsets) or
                    offsets[1] - offsets[0] != math.prod(shape) * width or
                    offsets[1] > size - 8 - length):
                raise ValueError(f'invalid safetensors descriptor: {name}')
            intervals.append(tuple(offsets))
            tensors[name] = dict(name=name, shape=shape, dtype=dtype, path=path, offset=8 + length + offsets[0])
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError('overlapping or noncontiguous safetensors data')
            cursor = end
        if cursor != size - 8 - length:
            raise ValueError('unindexed safetensors data')
        return tensors
    except (OSError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid VoiceCLAP safetensors: {path}') from error


def _checkpoint(source):
    single = source / 'model.safetensors'
    if single.exists():
        return _safetensors(single)
    index = _json(source, 'model.safetensors.index.json').get('weight_map')
    if not isinstance(index, dict) or not index:
        raise ValueError('invalid VoiceCLAP shard index')
    for filename in index.values():
        if (not isinstance(filename, str) or Path(filename).name != filename or
                '/' in filename or '\\' in filename or not filename.endswith('.safetensors')):
            raise ValueError('unsafe VoiceCLAP shard filename')
    tensors = {}
    for filename in sorted(set(index.values())):
        for name, descriptor in _safetensors(source / filename).items():
            if name in tensors or index.get(name) != filename:
                raise ValueError('conflicting VoiceCLAP shard index')
            tensors[name] = descriptor
    if set(tensors) != set(index):
        raise ValueError('incomplete VoiceCLAP shard index')
    return tensors


_QWEN_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def _validate_tokenizer(tokenizer, tokenizer_config):
    _require(tokenizer.get('model', {}), dict(type='BPE', dropout=None, unk_token=None,
             continuing_subword_prefix='', end_of_word_suffix='', fuse_unk=False,
             byte_fallback=False, ignore_merges=False), 'tokenizer BPE')
    _require(tokenizer, {'normalizer': {'type': 'NFC'}}, 'tokenizer normalization')
    expected = {'type': 'Sequence', 'pretokenizers': [
        {'type': 'Split', 'pattern': {'Regex': _QWEN_PATTERN}, 'behavior': 'Isolated', 'invert': False},
        {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False}]}
    _require(tokenizer, {'pre_tokenizer': expected}, 'tokenizer pre-tokenizer')
    bytelevel = {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False}
    _require(tokenizer, {'decoder': bytelevel, 'post_processor': bytelevel}, 'tokenizer postprocessing')
    _require(tokenizer_config, dict(tokenizer_class='Qwen2Tokenizer', add_prefix_space=False,
             split_special_tokens=False, bos_token=None, eos_token='<|im_end|>',
             pad_token='<|endoftext|>'), 'tokenizer config')
    required = {'<|im_start|>', '<|im_end|>', '<|audio_bos|>', '<|audio_eos|>', '<|AUDIO|>', '<|endoftext|>'}
    added = tokenizer.get('added_tokens', [])
    if not isinstance(added, list):
        raise ValueError('invalid VoiceCLAP added tokens')
    found = {}
    for item in added:
        if item.get('content') in required:
            _require(item, dict(special=True, normalized=False, lstrip=False, rstrip=False,
                     single_word=False), 'tokenizer special tokens')
            if type(item.get('id')) is not int or item['id'] < 0 or item['content'] in found:
                raise ValueError('invalid VoiceCLAP special token ID')
            found[item['content']] = item['id']
    if set(found) != required or len(set(found.values())) != len(required):
        raise ValueError('missing or duplicate VoiceCLAP special tokens')


def _audio_template_ids(tokenizer):
    """Use the checkpoint's exact tokenizer, including added-token semantics."""
    from tokenizers import Tokenizer
    engine = Tokenizer.from_str(json.dumps(tokenizer))
    engine.no_padding()
    engine.no_truncation()
    prefix = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_bos|>'
    suffix = '<|audio_eos|>\nSummarize the above audio in one word:<|im_end|>\n<|im_start|>assistant\n'
    return (engine.encode(prefix, add_special_tokens=False).ids,
            engine.encode(suffix, add_special_tokens=False).ids)


@ModelBase.register('voiceclap')
class VoiceClapModel(ModelBase):
    def write(self):
        config = _json(self.source, 'config.json')
        text, audio = _validate_config(config)
        preprocessor = _json(self.source, 'preprocessor_config.json')
        _require(preprocessor, dict(feature_extractor_type='WhisperFeatureExtractor', feature_size=128,
                 hop_length=160, n_fft=400, sampling_rate=16000, chunk_length=300,
                 n_samples=4800000, nb_max_frames=30000, dither=0.0, padding_side='right',
                 padding_value=0.0, return_attention_mask=True, processor_class='Qwen2_5OmniProcessor'), 'preprocessor')
        modules = _json(self.source, 'modules.json')
        expected = [('','sentence_transformers.base.modules.transformer.Transformer'),
                    ('1_Pooling','sentence_transformers.sentence_transformer.modules.pooling.Pooling'),
                    ('2_Normalize','sentence_transformers.sentence_transformer.modules.normalize.Normalize')]
        if (not isinstance(modules, list) or len(modules) != 3 or any(
                entry.get('idx') != i or entry.get('path') != path or entry.get('type') != kind
                for i, (entry, (path, kind)) in enumerate(zip(modules, expected)))):
            raise ValueError('unsupported VoiceCLAP modules (merged model with lasttoken/Normalize required)')
        pooling = _json(self.source, '1_Pooling/config.json')
        _require(pooling, dict(embedding_dimension=text['hidden_size'], pooling_mode='lasttoken', include_prompt=True), 'pooling')
        sentence_bert = _json(self.source, 'sentence_bert_config.json')
        _require(sentence_bert, dict(transformer_task='any-to-any', module_output_name='token_embeddings',
                 processing_kwargs={'chat_template': {'chat_template': 'sentence_transformers',
                                                      'add_generation_prompt': True}}), 'sentence_bert')
        modalities = sentence_bert.get('modality_config', {})
        for modality in ('text', 'audio', 'message'):
            _require(modalities.get(modality, {}), dict(method='forward', method_output_name=['hidden_states', -1]), 'modality')
        _require(modalities.get('message', {}), {'format': 'structured'}, 'message format')
        sentence_transformers = _json(self.source, 'config_sentence_transformers.json')
        _require(sentence_transformers, dict(model_type='SentenceTransformer', default_prompt_name='default', similarity_fn_name='cosine'), 'sentence_transformers')
        if not isinstance(sentence_transformers.get('prompts', {}).get('default'), str):
            raise ValueError('missing VoiceCLAP default system prompt')
        try:
            template = (self.source / 'additional_chat_templates/sentence_transformers.jinja').read_text(encoding='utf-8')
        except OSError as error:
            raise ValueError('missing VoiceCLAP sentence_transformers chat template') from error
        if hashlib.sha256(template.encode('utf-8')).hexdigest() != _TEMPLATE_SHA256:
            raise ValueError('unsupported VoiceCLAP sentence_transformers chat template')
        tokenizer, tokenizer_config = _json(self.source, 'tokenizer.json'), _json(self.source, 'tokenizer_config.json')
        _validate_tokenizer(tokenizer, tokenizer_config)
        tensors = _checkpoint(self.source)
        expected_shapes = _expected_shapes(text, audio)
        for name, shape in expected_shapes.items():
            if name not in tensors or tensors[name]['shape'] != shape:
                raise ValueError(f'missing or mismatched VoiceCLAP tensor: {name}')
        optional = {'audio_tower.audio_bos_eos_token.weight': [2, text['hidden_size']]}
        selected = []
        for name, descriptor in tensors.items():
            mapped = tensor_name(name)
            if mapped is None:
                continue
            if name not in expected_shapes:
                if name not in optional or descriptor['shape'] != optional[name]:
                    raise ValueError(f'unexpected VoiceCLAP tensor: {name}')
            selected.append(dict(descriptor, name=mapped))
        prefix, suffix = _audio_template_ids(tokenizer)
        metadata = {'general.architecture': 'voiceclap', 'general.name': 'VoiceCLAP-Large',
                    'voiceclap.chat_template': template,
                    'voiceclap.audio_prefix': json.dumps(prefix), 'voiceclap.audio_suffix': json.dumps(suffix)}
        for key, value in [('config', config), ('preprocessor', preprocessor), ('pooling', pooling),
                           ('tokenizer', tokenizer), ('tokenizer_config', tokenizer_config),
                           ('sentence_bert', sentence_bert), ('sentence_transformers', sentence_transformers)]:
            metadata['voiceclap.' + key] = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
        # Snapshot paths identify a real source; never label arbitrary local weights
        # with the official revision solely because their architecture matches.
        if self.source.parent.name == 'snapshots' and len(self.source.name) == 40 and all(
                c in '0123456789abcdef' for c in self.source.name):
            metadata['voiceclap.source_revision'] = self.source.name
        write_gguf_streaming(self.destination, selected, metadata, self.dtype)
