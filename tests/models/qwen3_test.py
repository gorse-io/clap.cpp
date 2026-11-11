"""No-download standard llama.cpp GGUF scalar, tokenizer, API and CLI contracts."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import wave
from qwen3_fixture import generate, model_data, standard_metadata, write_gguf

build, output = map(Path, sys.argv[1:])
generate(output)
for dtype in ['f32', 'f16']:
    subprocess.run([str(build / 'test-qwen3-model'), str(output / f'model-{dtype}.gguf'),
                    str(output / f'fixture-{dtype}.json')], check=True)
f = json.loads((output / 'fixture-f32.json').read_text())


def embedding(path, text='ab'):
    return subprocess.run([str(build / 'clap'), 'embedding', '--model', str(path), '--text', text],
                          capture_output=True, text=True)


for case in f['text_cases']:
    p = embedding(output / 'model-f32.gguf', case['text'])
    assert p.returncode == 0, p.stderr
    actual = json.loads(p.stdout)
    assert len(actual) == 32 and max(abs(a - b) for a, b in zip(actual, case['expected'])) < 3e-5
wav = output / 'unsupported.wav'
with wave.open(str(wav), 'wb') as audio:
    audio.setnchannels(1)
    audio.setsampwidth(2)
    audio.setframerate(16000)
    audio.writeframes(b'\0\0' * 32)
for extra in [[], ['--frames']]:
    p = subprocess.run([str(build / 'clap'), 'embedding', '--model', str(output / 'model-f32.gguf'),
                        '--audio', str(wav), *extra], capture_output=True, text=True)
    assert p.returncode != 0 and ('encoder' in p.stderr or 'frame' in p.stderr), p.stderr

c, t, tensors = model_data()
base = standard_metadata(c, t)
invalid = []
for key, value in [
    ('qwen3.attention.head_count_kv', 3), ('qwen3.attention.key_length', 3),
    ('qwen3.attention.value_length', 8), ('qwen3.rope.dimension_count', 8),
    ('qwen3.pooling_type', 1), ('qwen3.attention.causal', False),
    ('qwen3.rope.scaling.type', 'linear'), ('qwen3.attention.sliding_window', 32),
    ('qwen3.embedding_length', '32'), ('qwen3.block_count', 0),
    ('qwen3.rope.freq_base', -1.0), ('qwen3.attention.layer_norm_rms_epsilon', float('nan')),
    ('tokenizer.ggml.pre', 'unknown'), ('tokenizer.ggml.model', 'llama'),
    ('tokenizer.ggml.tokens', base['tokenizer.ggml.tokens'][:-1]),
    ('tokenizer.ggml.tokens', [1] * 266), ('tokenizer.ggml.token_type', ['x'] * 266),
    ('tokenizer.ggml.token_type', [99] * 266), ('tokenizer.ggml.merges', [1]),
    ('tokenizer.ggml.merges', ['missing token']), ('tokenizer.ggml.add_eos_token', 'true'),
    ('tokenizer.ggml.eos_token_id', 266), ('qwen3.config', '{}'),
]:
    invalid.append((key, dict(base, **{key: value}), tensors))
for key in ['qwen3.context_length', 'qwen3.attention.head_count', 'tokenizer.ggml.tokens']:
    changed = dict(base)
    del changed[key]
    invalid.append(('missing-' + key, changed, tensors))
# Old native-format GGUF must fail rather than falling back to embedded HF JSON.
invalid.append(('legacy', {'general.architecture': 'qwen3', 'qwen3.config': json.dumps(c),
                           'qwen3.tokenizer': json.dumps(t)}, tensors))
for mode in ['missing', 'wrong-shape', 'extra']:
    changed = copy.deepcopy(tensors)
    name = 'model.layers.0.self_attn.q_proj.weight'
    if mode == 'missing':
        del changed[name]
    elif mode == 'wrong-shape':
        changed[name] = ((1, 32), [0.0] * 32)
    else:
        changed['unexpected.weight'] = ((1,), [1.0])
    invalid.append((mode, base, changed))
for label, metadata, weights in invalid:
    path = output / f'invalid-{label}.gguf'
    write_gguf(path, metadata, weights)
    p = embedding(path)
    assert p.returncode != 0 and ('CLAP:' in p.stderr or 'json.exception' in p.stderr), (label, p.stderr)

# The GGUF flags, not an HF postprocessor, control BOS/EOS. No NFC is applied.
from qwen3_fixture import oracle
for bos, eos in [(False, False), (True, False), (True, True)]:
    meta = dict(base, **{'tokenizer.ggml.add_bos_token': bos, 'tokenizer.ggml.add_eos_token': eos})
    path = output / f'special-{bos}-{eos}.gguf'
    write_gguf(path, meta, tensors)
    expected = oracle(c, tensors, ([256] if bos else []) + [265] + ([256] if eos else []))
    p = embedding(path)
    assert p.returncode == 0, p.stderr
    assert max(abs(a - b) for a, b in zip(json.loads(p.stdout), expected)) < 3e-5
    if not bos and not eos:
        p = embedding(path, '')
        assert p.returncode != 0 and 'token count' in p.stderr, p.stderr

# Existing quantizers remain available, but Qwen3 is not a supported output architecture.
for preset in ['BF16', 'Q8_0', 'Q4_K_M']:
    path = output / f'not-supported-{preset}.gguf'
    path.unlink(missing_ok=True)
    p = subprocess.run([str(build / 'clap'), 'quantize', str(output / 'model-f32.gguf'),
                        str(path), '--type', preset], capture_output=True, text=True)
    assert p.returncode != 0 and 'unsupported architecture' in p.stderr and not path.exists(), p.stderr
print('Qwen3 standard GGUF scalar/API/CLI/tokenizer and invalid-contract regressions passed')
