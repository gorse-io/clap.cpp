"""Native Qwen3 embedding GGUF; bounded-memory HF safetensors conversion.

HF tensor names/layouts are retained, as in the other clap.cpp converters.
No torch or transformers dependency. The checkpoint's tokenizer.json is embedded.
"""

import json
import math
import os
from pathlib import Path
import struct
import tempfile

from .base import ModelBase, string
from .clap2 import _safetensors_header

LLAMA_REVISION = '202c1a229f5a83d1c9d758ed0f2d43760fa4bab0'
QWEN_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def validate_config(c):
    for k in (
        'hidden_size',
        'intermediate_size',
        'num_hidden_layers',
        'num_attention_heads',
        'num_key_value_heads',
        'head_dim',
        'vocab_size',
        'max_position_embeddings',
    ):
        if type(c.get(k)) is not int or not 0 < c[k] <= 2**20:
            raise ValueError(f'invalid Qwen3 {k}')
    if (
        c.get('model_type') != 'qwen3'
        or c.get('hidden_act') != 'silu'
        or c.get('attention_bias', False) is not False
        or c.get('use_sliding_window', False) is not False
        or c.get('rope_scaling') is not None
    ):
        raise ValueError(
            'unsupported Qwen3 architecture/activation/bias/sliding window/RoPE scaling'
        )
    if (
        c['num_attention_heads'] % c['num_key_value_heads']
        or c['head_dim'] % 2
        or c['num_attention_heads'] * c['head_dim'] > 2**20
    ):
        raise ValueError('invalid Qwen3 GQA/head dimension')
    for k in ('rms_norm_eps', 'rope_theta'):
        if type(c.get(k)) not in (int, float) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f'invalid Qwen3 {k}')
    for k in ('eos_token_id', 'bos_token_id', 'pad_token_id'):
        if (
            k in c
            and c[k] is not None
            and (type(c[k]) is not int or not 0 <= c[k] < c['vocab_size'])
        ):
            raise ValueError(f'invalid Qwen3 {k}')
    if type(c.get('eos_token_id')) is not int:
        raise ValueError('Qwen3 requires a terminal EOS id')


def validate_tokenizer(t, c):
    def expect(obj, fields):
        if not isinstance(obj, dict) or any(obj.get(k) != v for k, v in fields.items()):
            raise ValueError('unsupported Qwen3 tokenizer configuration')

    expect(t.get('normalizer'), dict(type='NFC'))
    pre = t.get('pre_tokenizer', {})
    expect(pre, dict(type='Sequence'))
    parts = pre.get('pretokenizers', [])
    if len(parts) != 2:
        raise ValueError('unsupported Qwen3 pretokenizer')
    expect(
        parts[0],
        dict(type='Split', pattern={'Regex': QWEN_REGEX}, behavior='Isolated', invert=False),
    )
    expect(parts[1], dict(type='ByteLevel', add_prefix_space=False, use_regex=False))
    expect(t.get('decoder'), dict(type='ByteLevel'))
    m = t.get('model', {})
    expect(
        m, dict(type='BPE', dropout=None, unk_token=None, byte_fallback=False, ignore_merges=False)
    )
    if m.get('continuing_subword_prefix') not in ('', None) or m.get('end_of_word_suffix') not in (
        '',
        None,
    ):
        raise ValueError('unsupported Qwen3 BPE affixes')
    vocab = m.get('vocab')
    merges = m.get('merges')
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise ValueError('invalid Qwen3 BPE')
    ids = dict(vocab)
    for a in t.get('added_tokens', []):
        expect(a, dict(single_word=False, lstrip=False, rstrip=False, normalized=False))
        if not isinstance(a.get('content'), str) or not a['content']:
            raise ValueError('invalid Qwen3 added token')
        ids[a['content']] = a['id']
    if len(ids) != c['vocab_size'] or set(ids.values()) != set(range(c['vocab_size'])):
        raise ValueError('Qwen3 tokenizer vocabulary mismatch')
    if ids.get('<|endoftext|>') != c['eos_token_id']:
        raise ValueError('Qwen3 tokenizer EOS mismatch')
    post = t.get('post_processor', {})
    expect(post, dict(type='Sequence'))
    processors = post.get('processors', [])
    if len(processors) != 2:
        raise ValueError('unsupported Qwen3 postprocessor')
    expect(processors[0], dict(type='ByteLevel', add_prefix_space=False, use_regex=False))
    expect(
        processors[1],
        dict(
            type='TemplateProcessing',
            single=[
                {'Sequence': {'id': 'A', 'type_id': 0}},
                {'SpecialToken': {'id': '<|endoftext|>', 'type_id': 0}},
            ],
        ),
    )
    spec = processors[1].get('special_tokens', {}).get('<|endoftext|>', {})
    expect(spec, dict(ids=[c['eos_token_id']], tokens=['<|endoftext|>']))
    # Require all bytes and well-formed, realizable merge pairs.
    regular = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_chars = {b: chr(b) for b in regular}
    byte_chars.update(
        {b: chr(256 + i) for i, b in enumerate(b for b in range(256) if b not in regular)}
    )
    if any(ch not in vocab for ch in byte_chars.values()):
        raise ValueError('Qwen3 missing byte token')
    available = set(byte_chars.values())
    for merge in merges:
        pair = merge.split(' ') if isinstance(merge, str) else merge
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(v not in available for v in pair)
            or ''.join(pair) not in vocab
        ):
            raise ValueError('invalid Qwen3 merge')
        available.add(''.join(pair))


def expected_tensor_shapes(c):
    w = c['hidden_size']
    d = c['head_dim']
    q = c['num_attention_heads'] * d
    kv = c['num_key_value_heads'] * d
    f = c['intermediate_size']
    result = {'model.embed_tokens.weight': (c['vocab_size'], w), 'model.norm.weight': (w,)}
    for i in range(c['num_hidden_layers']):
        p = f'model.layers.{i}.'
        for n in ('input_layernorm', 'post_attention_layernorm'):
            result[p + n + '.weight'] = (w,)
        for n in ('q_norm', 'k_norm'):
            result[p + 'self_attn.' + n + '.weight'] = (d,)
        for n, out, inp in [
            ('q_proj', q, w),
            ('k_proj', kv, w),
            ('v_proj', kv, w),
            ('o_proj', w, q),
        ]:
            result[p + 'self_attn.' + n + '.weight'] = (out, inp)
        for n, out, inp in [('gate_proj', f, w), ('up_proj', f, w), ('down_proj', w, f)]:
            result[p + 'mlp.' + n + '.weight'] = (out, inp)
    return result


def write_qwen3(destination, checkpoint, metadata, c, dtype):
    """Validate all offsets/shapes first, then atomically stream <=4 MiB chunks."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if dtype not in ('f32', 'f16'):
        raise ValueError('unsupported Qwen3 output dtype')
    with checkpoint.open('rb') as source:
        tensors, data_start = _safetensors_header(source)
        # Sentence Transformers Qwen3Model exports omit the CausalLM 'model.' prefix.
        if 'embed_tokens.weight' in tensors:
            tensors = {
                ('model.' + name if name != 'lm_head.weight' else name): entry
                for name, entry in tensors.items()
            }
        expected = expected_tensor_shapes(c)
        # A causal-LM export may carry an unused output head; embeddings never use it.
        if 'lm_head.weight' in tensors:
            expected['lm_head.weight'] = (c['vocab_size'], c['hidden_size'])
        if set(tensors) != set(expected):
            raise ValueError(
                f'Qwen3 tensor schema mismatch: missing={set(expected)-set(tensors)}, unexpected={set(tensors)-set(expected)}'
            )
        encoded = []
        ranges = []
        for name, t in sorted(tensors.items()):
            shape = t.get('shape')
            typ = t.get('dtype')
            offsets = t.get('data_offsets')
            if (
                not isinstance(shape, list)
                or tuple(shape) != expected[name]
                or any(type(d) is not int or d <= 0 for d in shape)
            ):
                raise ValueError(f'Qwen3 tensor shape: {name}')
            if typ not in ('F32', 'F16', 'BF16'):
                raise ValueError(f'Qwen3 unsupported storage: {typ}')
            count = math.prod(shape)
            size = 4 if typ == 'F32' else 2
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
                or offsets[1] - offsets[0] != count * size
            ):
                raise ValueError(f'Qwen3 offsets: {name}')
            ranges.append(tuple(offsets))
            if name != 'lm_head.weight':
                encoded.append(
                    (name, shape, typ, offsets[0], count, dtype == 'f16' and len(shape) == 2)
                )
        end = 0
        for start, stop in sorted(ranges):
            if start != end:
                raise ValueError('Qwen3 overlapping/gapped tensor offsets')
            end = stop
        if data_start + end != os.fstat(source.fileno()).st_size:
            raise ValueError('Qwen3 safetensors file size mismatch')
        header = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(encoded), len(metadata)))
        for k, v in metadata.items():
            header += string(k) + struct.pack('<I', 8) + string(v)
        offset = 0
        for name, shape, _, _, count, half in encoded:
            if len(name.encode()) >= 64:
                raise ValueError('Qwen3 tensor name too long')
            header += (
                string(name)
                + struct.pack('<I', len(shape))
                + struct.pack('<' + 'Q' * len(shape), *reversed(shape))
                + struct.pack('<IQ', int(half), offset)
            )
            offset += (count * (2 if half else 4) + 31) // 32 * 32
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as out:
                out.write(header)
                out.write(b'\0' * (-len(header) % 32))
                for name, _, typ, start, count, half in encoded:
                    source.seek(data_start + start)
                    remaining = count
                    while remaining:
                        n = min(remaining, 1024 * 1024)
                        size = 4 if typ == 'F32' else 2
                        chunk = source.read(n * size)
                        if len(chunk) != n * size:
                            raise ValueError(f'truncated Qwen3 tensor {name}')
                        if np is None:
                            if typ == 'BF16':
                                values = struct.unpack(
                                    '<' + 'f' * n,
                                    b''.join(
                                        b'\0\0' + chunk[i : i + 2] for i in range(0, len(chunk), 2)
                                    ),
                                )
                            else:
                                values = struct.unpack(
                                    '<' + ('f' if typ == 'F32' else 'e') * n, chunk
                                )
                            if not all(math.isfinite(v) for v in values):
                                raise ValueError(f'nonfinite Qwen3 tensor {name}')
                            data = struct.pack('<' + ('e' if half else 'f') * n, *values)
                        else:
                            values = (
                                (np.frombuffer(chunk, dtype='<u2').astype('<u4') << 16).view('<f4')
                                if typ == 'BF16'
                                else np.frombuffer(chunk, dtype='<f4' if typ == 'F32' else '<f2')
                            )
                            if not np.isfinite(values).all():
                                raise ValueError(f'nonfinite Qwen3 tensor {name}')
                            with np.errstate(over='ignore'):
                                values = values.astype('<f2' if half else '<f4')
                            if not np.isfinite(values).all():
                                raise ValueError(f'Qwen3 output overflow {name}')
                            data = values.tobytes()
                        out.write(data)
                        remaining -= n
                    out.write(b'\0' * (-count * (2 if half else 4) % 32))
            os.replace(temp, destination)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    print(f'{destination}: {len(encoded)} tensors ({dtype}, qwen3)')


@ModelBase.register('qwen3')
class Qwen3Model(ModelBase):
    def write(self):
        c = json.loads((self.source / 'config.json').read_text(encoding='utf-8'))
        validate_config(c)
        path = self.options.get('tokenizer') or self.source / 'tokenizer.json'
        t = json.loads(Path(path).read_text(encoding='utf-8'))
        validate_tokenizer(t, c)
        metadata = {
            'general.architecture': 'qwen3',
            'general.name': 'Qwen3 Embedding',
            'qwen3.config': json.dumps(c, separators=(',', ':')),
            'qwen3.tokenizer': json.dumps(t, separators=(',', ':'), ensure_ascii=False),
            'qwen3.llama_reference_revision': LLAMA_REVISION,
        }
        write_qwen3(self.destination, self.source / 'model.safetensors', metadata, c, self.dtype)
