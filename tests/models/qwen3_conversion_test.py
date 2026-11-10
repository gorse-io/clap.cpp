"""Real conversion of tiny BF16/F16/F32 safetensors, no ML packages/downloads."""

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3_fixture import model_data
from conversion import get_model_architecture, get_model_class


def read_gguf(path):
    with path.open('rb') as f:

        def unpack(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))

        def string():
            return f.read(unpack('Q')[0]).decode()

        assert f.read(4) == b'GGUF'
        version, nt, nm = unpack('IQQ')
        assert version == 3
        metadata = {}
        for _ in range(nm):
            key = string()
            assert unpack('I')[0] == 8
            metadata[key] = string()
        descriptors = []
        for _ in range(nt):
            name = string()
            ndim = unpack('I')[0]
            dims = unpack('Q' * ndim)
            typ, offset = unpack('IQ')
            descriptors.append((name, tuple(reversed(dims)), typ, offset))
        start = (f.tell() + 31) // 32 * 32
        result = {}
        for name, shape, typ, offset in descriptors:
            f.seek(start + offset)
            fmt = 'e' if typ else 'f'
            import math

            result[name] = (shape, unpack(fmt * math.prod(shape)))
        return metadata, result


c, t, tensors = model_data()
with tempfile.TemporaryDirectory() as tmp:
    source = Path(tmp)
    (source / 'config.json').write_text(json.dumps(c))
    (source / 'tokenizer.json').write_text(json.dumps(t))
    assert get_model_architecture(source) == 'qwen3'
    cls = get_model_class('qwen3')
    for dtype, fmt in [('F32', 'f'), ('F16', 'e'), ('BF16', None)]:
        header = {}
        body = bytearray()
        for name, (shape, values) in tensors.items():
            start = len(body)
            for v in values:
                body.extend(struct.pack('<' + fmt, v) if fmt else struct.pack('<f', v)[2:])
            header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(body)])
        data = json.dumps(header).encode()
        (source / 'model.safetensors').write_bytes(struct.pack('<Q', len(data)) + data + body)
        for out_dtype in ['f32', 'f16']:
            output = source / f'{dtype}-{out_dtype}.gguf'
            cls(source, output, out_dtype).write()
            metadata, converted = read_gguf(output)
            assert metadata['general.architecture'] == 'qwen3'
            assert json.loads(metadata['qwen3.tokenizer']) == t
            assert set(converted) == set(tensors)
            for name, (shape, actual) in converted.items():
                assert shape == tensors[name][0]
                for a, original in zip(actual, tensors[name][1]):
                    reference = original
                    if dtype == 'BF16':
                        reference = struct.unpack('<f', b'\0\0' + struct.pack('<f', reference)[2:])[
                            0
                        ]
                    elif dtype == 'F16':
                        reference = struct.unpack('<e', struct.pack('<e', reference))[0]
                    if out_dtype == 'f16' and len(shape) == 2:
                        reference = struct.unpack('<e', struct.pack('<e', reference))[0]
                    assert a == reference, (name, a, reference)
    # The real unsloth export has bare Qwen3Model names, despite CausalLM config.
    bare = {name.removeprefix('model.'): entry for name, entry in header.items()}
    data = json.dumps(bare).encode()
    (source / 'model.safetensors').write_bytes(struct.pack('<Q', len(data)) + data + body)
    output = source / 'bare.gguf'
    cls(source, output, 'f32').write()
    assert set(read_gguf(output)[1]) == set(tensors)
    # UTF-8 checkpoint JSON must not depend on the process locale (e.g. Windows).
    import os
    import subprocess

    unicode_config = dict(c, _source_description='\u4e2d\u6587')
    unicode_tokenizer = dict(t, _source_description='\u4e2d\u6587')
    (source / 'config.json').write_text(
        json.dumps(unicode_config, ensure_ascii=False), encoding='utf-8'
    )
    (source / 'tokenizer.json').write_text(
        json.dumps(unicode_tokenizer, ensure_ascii=False), encoding='utf-8'
    )
    subprocess.run(
        [
            sys.executable, '-X', 'utf8=0', '-c',
            'from pathlib import Path; from conversion import get_model_class; '
            'import sys; get_model_class("qwen3")(Path(sys.argv[1]), '
            'Path(sys.argv[2]), "f32").write()',
            str(source), str(source / 'locale.gguf'),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, LC_ALL='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0'),
        check=True,
    )
    locale_metadata, _ = read_gguf(source / 'locale.gguf')
    assert json.loads(locale_metadata['qwen3.tokenizer']) == unicode_tokenizer
    (source / 'config.json').write_text(json.dumps(c), encoding='utf-8')
    (source / 'tokenizer.json').write_text(json.dumps(t), encoding='utf-8')
    # Validate the safetensors schema and offsets before publishing any output.
    original = output.read_bytes()
    for field, value in [('dtype', 'I32'), ('shape', [1]), ('data_offsets', [0, 1])]:
        changed = copy.deepcopy(bare)
        changed['embed_tokens.weight'][field] = value
        raw = json.dumps(changed).encode()
        (source / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + body)
        try:
            cls(source, output, 'f32').write()
        except ValueError:
            pass
        else:
            raise AssertionError(field)
        assert output.read_bytes() == original
    # A payload NaN is caught during streaming; atomic replacement preserves output.
    raw = json.dumps(bare).encode()
    nonfinite = bytearray(body)
    nonfinite[:2] = b'\xc0\x7f'  # BF16 quiet NaN
    (source / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + nonfinite)
    try:
        cls(source, output, 'f32').write()
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite payload')
    assert output.read_bytes() == original
    from conversion.qwen3 import validate_config, validate_tokenizer

    validate_config(c)
    validate_tokenizer(t, c)
    for key, value in [
        ('num_key_value_heads', 3),
        ('head_dim', 3),
        ('hidden_act', 'gelu'),
        ('rope_scaling', {'rope_type': 'linear', 'factor': 2}),
        ('attention_bias', True),
        ('use_sliding_window', True),
        ('rms_norm_eps', float('nan')),
    ]:
        bad = dict(c, **{key: value})
        try:
            validate_config(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(key)
    for key, value in [
        ('normalizer', None),
        ('post_processor', None),
        ('pre_tokenizer', {'type': 'ByteLevel'}),
    ]:
        bad = copy.deepcopy(t)
        bad[key] = value
        try:
            validate_tokenizer(bad, c)
        except ValueError:
            pass
        else:
            raise AssertionError(key)
print('Qwen3 conversion contracts passed')
