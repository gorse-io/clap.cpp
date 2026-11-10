"""Default no-download scalar oracle, API, CLI and storage regressions."""

import json
import copy
import math
from pathlib import Path
import subprocess
import sys
import struct
from qwen3_fixture import generate, model_data, write_gguf, oracle


def gguf_layout(f):
    def unpack(fmt):
        return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))

    def string():
        return f.read(unpack('Q')[0]).decode()

    assert f.read(4) == b'GGUF'
    version, nt, nm = unpack('IQQ')
    assert version == 3
    alignment = 32
    for _ in range(nm):
        key = string()
        (typ,) = unpack('I')
        if typ == 8:
            string()
        else:
            assert typ == 4
            (value,) = unpack('I')
            if key == 'general.alignment':
                alignment = value
    entries = []
    for _ in range(nt):
        name = string()
        (ndim,) = unpack('I')
        shape = tuple(reversed(unpack('Q' * ndim)))
        while len(shape) > 1 and shape[0] == 1:
            shape = shape[1:]
        typ, offset = unpack('IQ')
        entries.append((name, shape, typ, offset))
    return entries, (f.tell() + alignment - 1) // alignment * alignment


def tensor_types(path):
    with path.open('rb') as f:
        entries, _ = gguf_layout(f)
        return {name: typ for name, _, typ, _ in entries}


def stored_oracle_weights(path):
    # Reuse the repository's canonical storage exporter. The decoder math below
    # remains independent scalar Python; K byte layouts are tested by test-q8.
    decoded = path.with_suffix('.decoded.gguf')
    subprocess.run([str(build / 'test-q8'), 'decode', str(path), str(decoded)], check=True)
    with decoded.open('rb') as f:
        entries, start = gguf_layout(f)
        result = {}
        for name, shape, typ, offset in entries:
            assert typ == 0
            count = math.prod(shape)
            f.seek(start + offset)
            result[name] = (shape, struct.unpack('<' + 'f' * count, f.read(count * 4)))
        return result


build, output = map(Path, sys.argv[1:])
generate(output)
for dtype in ['f32', 'f16']:
    subprocess.run(
        [
            str(build / 'test-qwen3-model'),
            str(output / f'model-{dtype}.gguf'),
            str(output / f'fixture-{dtype}.json'),
        ],
        check=True,
    )
f = json.loads((output / 'fixture-f32.json').read_text())
for case in f['text_cases']:
    p = subprocess.run(
        [
            str(build / 'clap'),
            'embedding',
            '--model',
            str(output / 'model-f32.gguf'),
            '--text',
            case['text'],
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    actual = json.loads(p.stdout)
    assert len(actual) == 32 and max(abs(a - b) for a, b in zip(actual, case['expected'])) < 3e-5
# CLI audio/frame calls must fail through the same model capability contract.
import wave

wav = output / 'unsupported.wav'
with wave.open(str(wav), 'wb') as audio:
    audio.setnchannels(1)
    audio.setsampwidth(2)
    audio.setframerate(16000)
    audio.writeframes(b'\0\0' * 32)
for extra in [[], ['--frames']]:
    run = subprocess.run(
        [
            str(build / 'clap'),
            'embedding',
            '--model',
            str(output / 'model-f32.gguf'),
            '--audio',
            str(wav),
            *extra,
        ],
        text=True,
        capture_output=True,
    )
    assert run.returncode != 0 and ('encoder' in run.stderr or 'frame' in run.stderr), run.stderr
tiny_c, _, _ = model_data()
for preset in ['BF16', 'Q8_0']:
    path = output / f'model-{preset}.gguf'
    path.unlink(missing_ok=True)
    subprocess.run(
        [
            str(build / 'clap'),
            'quantize',
            str(output / 'model-f32.gguf'),
            str(path),
            '--type',
            preset,
        ],
        check=True,
    )
    stored = tensor_types(path)
    assert stored['model.embed_tokens.weight'] == {'BF16': 30, 'Q8_0': 8}[preset]
    assert stored['model.norm.weight'] == 0
    rounded = stored_oracle_weights(path)
    changed = copy.deepcopy(f)
    changed['tolerance'] = 3e-5
    for case in changed['text_cases'] + changed['token_cases']:
        case['expected'] = oracle(tiny_c, rounded, case['tokens'])
    fixture = output / f'fixture-{preset}.json'
    fixture.write_text(json.dumps(changed))
    subprocess.run([str(build / 'test-qwen3-model'), str(path), str(fixture)], check=True)
print('Qwen3 native scalar/API/CLI/storage regressions passed')

c, t, tensors = model_data()
base = {
    'general.architecture': 'qwen3',
    'qwen3.config': json.dumps(c),
    'qwen3.tokenizer': json.dumps(t),
}
invalid = []
for key, value in [
    ('num_key_value_heads', 3),
    ('head_dim', 3),
    ('attention_bias', True),
    ('rope_scaling', {'factor': 2}),
    ('hidden_act', 'gelu'),
]:
    invalid.append(
        (key, dict(base, **{'qwen3.config': json.dumps(dict(c, **{key: value}))}), tensors)
    )
for key, value in [
    ('normalizer', None),
    ('post_processor', None),
    ('pre_tokenizer', {'type': 'ByteLevel'}),
]:
    bad = copy.deepcopy(t)
    bad[key] = value
    invalid.append(
        ('tokenizer-' + key, dict(base, **{'qwen3.tokenizer': json.dumps(bad)}), tensors)
    )
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
    result = subprocess.run(
        [str(build / 'clap'), 'embedding', '--model', str(path), '--text', 'x'],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and (
        'Qwen3' in result.stderr or 'tensor' in result.stderr or 'json.exception' in result.stderr
    ), (label, result.stderr)
print('Qwen3 invalid metadata/tensors rejected')

# K-quant blocks need 256-wide rows; keep this separate from the tiny oracle.
large_c, large_tok, large_weights = model_data(256)
large_meta = {
    'general.architecture': 'qwen3',
    'qwen3.config': json.dumps(large_c),
    'qwen3.tokenizer': json.dumps(large_tok),
}
large_source = output / 'k-source.gguf'
write_gguf(large_source, large_meta, large_weights)
large_target = output / 'model-Q4_K_M.gguf'
large_target.unlink(missing_ok=True)
subprocess.run(
    [str(build / 'clap'), 'quantize', str(large_source), str(large_target), '--type', 'Q4_K_M'],
    check=True,
)
stored = tensor_types(large_target)
assert {12, 14}.issubset(stored.values())
assert stored['model.layers.0.self_attn.v_proj.weight'] == 14
assert stored['model.layers.0.mlp.down_proj.weight'] == 14
assert stored['model.norm.weight'] == 0
rounded = stored_oracle_weights(large_target)
expected = oracle(large_c, rounded, [265, 256])
large_fixture = dict(
    sample_rate=0,
    projection_dim=256,
    context_length=128,
    tolerance=3e-5,
    text_cases=[dict(text='ab', tokens=[265, 256], expected=expected)],
    token_cases=[],
    waveform_cases=[],
)
large_json = output / 'fixture-Q4_K_M.json'
large_json.write_text(json.dumps(large_fixture))
subprocess.run([str(build / 'test-qwen3-model'), str(large_target), str(large_json)], check=True)
run = subprocess.run(
    [str(build / 'clap'), 'embedding', '--model', str(large_target), '--text', 'ab'],
    text=True,
    capture_output=True,
)
assert run.returncode == 0, run.stderr
actual = json.loads(run.stdout)
import math

assert len(actual) == 256 and abs(sum(v * v for v in actual) - 1) < 2e-5
cosine = sum(a * b for a, b in zip(actual, expected))
assert cosine > 0.999, cosine
print('Qwen3 Q4_K_M native inference cosine:', cosine)
