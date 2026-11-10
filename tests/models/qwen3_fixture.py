"""Deterministic, stdlib-only Qwen3 fixture and independent scalar decoder oracle."""

import json
import math
from pathlib import Path
import random
import struct
from clap2_fixture import byte_encoder, write_gguf

QWEN_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def model_data(width=32):
    c = dict(
        model_type='qwen3',
        hidden_size=width,
        intermediate_size=width * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=width // 2,
        vocab_size=266,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act='silu',
        attention_bias=False,
        use_sliding_window=False,
        rope_scaling=None,
        bos_token_id=256,
        eos_token_id=256,
        pad_token_id=257,
        tie_word_embeddings=True,
    )
    enc = byte_encoder()
    vocab = {v: k for k, v in enc.items()}
    merges = [('1', '2'), ("'", 'R'), ("'R", 'E'), ('_', 'a'), (enc[10], enc[10]), ('a', 'b')]
    for i, (a, b) in enumerate(merges, 260):
        vocab[a + b] = i
    added = [
        dict(
            id=i,
            content=s,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=True,
        )
        for i, s in enumerate(['<|endoftext|>', '<|pad|>', '<|im_start|>', '<|im_end|>'], 256)
    ]
    vocab.update({a['content']: a['id'] for a in added})
    tok = dict(
        version='1.0',
        normalizer=dict(type='NFC'),
        added_tokens=added,
        pre_tokenizer=dict(
            type='Sequence',
            pretokenizers=[
                dict(
                    type='Split', pattern=dict(Regex=QWEN_REGEX), behavior='Isolated', invert=False
                ),
                dict(type='ByteLevel', add_prefix_space=False, use_regex=False, trim_offsets=False),
            ],
        ),
        post_processor=dict(
            type='Sequence',
            processors=[
                dict(type='ByteLevel', add_prefix_space=False, use_regex=False, trim_offsets=False),
                dict(
                    type='TemplateProcessing',
                    single=[
                        {'Sequence': {'id': 'A', 'type_id': 0}},
                        {'SpecialToken': {'id': '<|endoftext|>', 'type_id': 0}},
                    ],
                    special_tokens={
                        '<|endoftext|>': dict(
                            id='<|endoftext|>', ids=[256], tokens=['<|endoftext|>']
                        )
                    },
                ),
            ],
        ),
        decoder=dict(type='ByteLevel', add_prefix_space=False, use_regex=False, trim_offsets=False),
        model=dict(
            type='BPE',
            vocab=vocab,
            merges=[list(m) for m in merges],
            dropout=None,
            unk_token=None,
            continuing_subword_prefix='',
            end_of_word_suffix='',
            fuse_unk=False,
            byte_fallback=False,
            ignore_merges=False,
        ),
    )
    rng = random.Random(571)
    tensors = {}

    def add(name, shape, norm=False):
        values = [
            (0.7 + rng.random() * 0.6) if norm else rng.uniform(-0.2, 0.2)
            for _ in range(math.prod(shape))
        ]
        tensors[name] = (shape, [struct.unpack('<f', struct.pack('<f', v))[0] for v in values])

    add('model.embed_tokens.weight', (266, width))
    add('model.norm.weight', (width,), True)
    d = c['head_dim']
    for i in range(2):
        p = f'model.layers.{i}.'
        for n in ('input_layernorm', 'post_attention_layernorm'):
            add(p + n + '.weight', (width,), True)
        for n in ('q_norm', 'k_norm'):
            add(p + 'self_attn.' + n + '.weight', (d,), True)
        for n, out, inp in [
            ('q_proj', 4 * d, width),
            ('k_proj', 2 * d, width),
            ('v_proj', 2 * d, width),
            ('o_proj', width, 4 * d),
        ]:
            add(p + 'self_attn.' + n + '.weight', (out, inp))
        for n, out, inp in [
            ('gate_proj', width * 2, width),
            ('up_proj', width * 2, width),
            ('down_proj', width, width * 2),
        ]:
            add(p + 'mlp.' + n + '.weight', (out, inp))
    return c, tok, tensors


def oracle(c, tensors, ids, *, causal=True, qknorm=True, rope=True, gqa=True, last=True):
    # Padding is a masked key, and pooling selects the final nonpadding query.
    w = c['hidden_size']
    d = c['head_dim']
    heads = c['num_attention_heads']
    kv = c['num_key_value_heads']

    def weight(n):
        return tensors[n + '.weight'][1]

    def linear(x, n):
        shape, a = tensors[n + '.weight']
        return [sum(v * a[j * len(x) + k] for k, v in enumerate(x)) for j in range(shape[0])]

    def norm(x, n):
        a = weight(n)
        scale = 1 / math.sqrt(sum(v * v for v in x) / len(x) + c['rms_norm_eps'])
        return [v * scale * a[k] for k, v in enumerate(x)]

    def rotate(x, pos):
        y = x[:]
        if rope:
            for k in range(d // 2):
                angle = pos * c['rope_theta'] ** (-2 * k / d)
                co = math.cos(angle)
                si = math.sin(angle)
                y[k] = x[k] * co - x[k + d // 2] * si
                y[k + d // 2] = x[k + d // 2] * co + x[k] * si
        return y

    x = [weight('model.embed_tokens')[i * w : (i + 1) * w] for i in ids]
    positions = [
        sum(t != c.get('pad_token_id', -1) for t in ids[: i + 1]) - 1 for i in range(len(ids))
    ]
    for layer in range(c['num_hidden_layers']):
        p = f'model.layers.{layer}.'
        normalized = [norm(row, p + 'input_layernorm') for row in x]
        qkv = []
        for name, nh in [('q', heads), ('k', kv), ('v', kv)]:
            allrows = []
            for i, row in enumerate(normalized):
                vec = linear(row, p + 'self_attn.' + name + '_proj')
                parts = [vec[h * d : (h + 1) * d] for h in range(nh)]
                if name != 'v':
                    parts = [
                        rotate(
                            norm(part, p + 'self_attn.' + name + '_norm') if qknorm else part,
                            positions[i],
                        )
                        for part in parts
                    ]
                allrows.append(parts)
            qkv.append(allrows)
        q, k, v = qkv
        att = []
        for i in range(len(ids)):
            row = []
            for h in range(heads):
                kh = h // (heads // kv) if gqa else h % kv
                scores = [
                    (
                        sum(a * b for a, b in zip(q[i][h], k[j][kh])) / math.sqrt(d)
                        if (not causal or j <= i) and ids[j] != c.get('pad_token_id', -1)
                        else -math.inf
                    )
                    for j in range(len(ids))
                ]
                top = max(scores)
                if top == -math.inf:
                    row.extend([0.0] * d)
                    continue
                probs = [math.exp(s - top) for s in scores]
                total = sum(probs)
                row.extend(
                    [sum(probs[j] * v[j][kh][z] for j in range(len(ids))) / total for z in range(d)]
                )
            att.append(linear(row, p + 'self_attn.o_proj'))
        x = [[a + b for a, b in zip(row, res)] for row, res in zip(x, att)]
        for i, row in enumerate(x):
            z = norm(row, p + 'post_attention_layernorm')
            gate = linear(z, p + 'mlp.gate_proj')
            up = linear(z, p + 'mlp.up_proj')
            down = linear(
                [a / (1 + math.exp(-a)) * b for a, b in zip(gate, up)], p + 'mlp.down_proj'
            )
            x[i] = [a + b for a, b in zip(row, down)]
    rows = [
        norm(row, 'model.norm') for i, row in enumerate(x) if ids[i] != c.get('pad_token_id', -1)
    ]
    pooled = rows[-1] if last else [sum(col) / len(rows) for col in zip(*rows)]
    scale = math.sqrt(sum(v * v for v in pooled))
    return [v / scale for v in pooled]


def generate(output, width=32):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c, t, tensors = model_data(width)
    metadata = {
        'general.architecture': 'qwen3',
        'qwen3.config': json.dumps(c),
        'qwen3.tokenizer': json.dumps(t),
    }
    texts = [
        ('', [256]),
        ('<|endoftext|>', [256, 256]),
        ('ab12', [265, 49, 50, 256]),
        ("I'RE_a\n\n", [73, 262, 263, 264, 256]),
        ('e\u0301 中文🙂', list('é 中文🙂'.encode()) + [256]),
        ('<|im_start|>ab<|im_end|>', [258, 265, 259, 256]),
        ('<|im_start|><|im_end|>', [258, 259, 256]),
    ]
    cases = [dict(text=s, tokens=ids, expected=oracle(c, tensors, ids)) for s, ids in texts]
    token_ids = [
        [3, 7, 9, 256],
        [257, 257, 3, 7, 9, 256],
        [3, 7, 9, 256, 257, 257],
        [3, 257, 7, 9, 256],
        [9, 7, 3, 256],
        [3],
        [3, 7],
        [3, 7, 9],
    ]
    fixture = dict(
        sample_rate=0,
        projection_dim=width,
        context_length=128,
        tolerance=3e-5,
        text_cases=cases,
        token_cases=[
            dict(name=str(ids), tokens=ids, expected=oracle(c, tensors, ids)) for ids in token_ids
        ],
        waveform_cases=[],
    )
    # Prove this nondegenerate fixture detects each architecture mistake.
    ref = oracle(c, tensors, [3, 7, 9, 256])
    for toggle in ['causal', 'qknorm', 'rope', 'gqa', 'last']:
        wrong = oracle(c, tensors, [3, 7, 9, 256], **{toggle: False})
        assert max(abs(a - b) for a, b in zip(ref, wrong)) > 1e-4, toggle
    for half in [False, True]:
        name = 'f16' if half else 'f32'
        write_gguf(output / f'model-{name}.gguf', metadata, tensors, half)
        rounded = {
            n: (
                shape,
                (
                    [struct.unpack('<e', struct.pack('<e', v))[0] for v in values]
                    if half and len(shape) == 2
                    else values
                ),
            )
            for n, (shape, values) in tensors.items()
        }
        f = json.loads(json.dumps(fixture))
        f['tolerance'] = 8e-4 if half else 3e-5
        for case in f['text_cases'] + f['token_cases']:
            case['expected'] = oracle(c, rounded, case['tokens'])
        (output / f'fixture-{name}.json').write_text(json.dumps(f))
    return metadata, tensors


if __name__ == '__main__':
    import sys

    generate(sys.argv[1])
