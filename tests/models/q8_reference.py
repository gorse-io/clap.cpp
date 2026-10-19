"""Small independent GGUF reader for storage-rounded Transformers oracles.

Q8 decoding is test-only; production conversion uses ggml's canonical quantizer.
"""
import struct
import numpy as np
import torch


def tensors(path):
    with open(path, 'rb') as stream:
        def unpack(fmt):
            return struct.unpack('<' + fmt, stream.read(struct.calcsize('<' + fmt)))
        def string():
            return stream.read(unpack('Q')[0]).decode('utf-8')
        def value(typ):
            if typ == 8:
                return string()
            if typ == 9:
                subtype, length = unpack('IQ')
                return [value(subtype) for _ in range(length)]
            return unpack({0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}[typ])[0]
        assert stream.read(4) == b'GGUF'
        version, count, keys = unpack('IQQ')
        assert version == 3
        metadata = {}
        for _ in range(keys):
            key = string()
            metadata[key] = value(unpack('I')[0])
        entries = []
        for _ in range(count):
            name = string()
            shape = unpack('Q' * unpack('I')[0])
            typ, offset = unpack('IQ')
            entries.append((name, tuple(reversed(shape)), typ, offset))
        alignment = metadata.get('general.alignment', 32)
        base = (stream.tell() + alignment - 1) // alignment * alignment
        for name, shape, typ, offset in entries:
            stream.seek(base + offset)
            size = int(np.prod(shape))
            if typ == 8:
                blocks = np.frombuffer(stream.read(size // 32 * 34), dtype=np.dtype([('d', '<f2'), ('q', 'i1', (32,))]))
                data = (blocks['d'].astype(np.float32)[:, None] * blocks['q']).reshape(shape)
            else:
                dt = np.dtype({0: '<f4', 1: '<f2'}[typ])
                data = np.frombuffer(stream.read(size * dt.itemsize), dtype=dt).astype(np.float32).reshape(shape)
            yield name, typ, data


def load_storage(model, path, name_map):
    """Install stored values and emulate ggml's per-linear activation packing."""
    stored = {name: (typ, data) for name, typ, data in tensors(path)}
    for handle in getattr(model, '_q8_oracle_hooks', []):
        handle.remove()
    model._q8_oracle_hooks = []
    def pack_input(module, args):
        x = args[0]
        if module._gguf_type == 1:
            return (x.half().float(), *args[1:])
        blocks = x.reshape(*x.shape[:-1], -1, 32)
        scale = blocks.abs().amax(dim=-1, keepdim=True) / 127
        inverse = torch.where(scale == 0, torch.zeros_like(scale), 1 / scale)
        # SIMD ggml activation packing rounds nearest-even and stores FP16 scales.
        packed = torch.round(blocks * inverse).clamp(-127, 127)
        return ((packed * scale.half().float()).reshape(x.shape), *args[1:])
    for name, module in model.named_modules():
        entry = stored.get(name_map(name + '.weight'))
        if isinstance(module, torch.nn.Linear) and entry and entry[0] in (1, 8):
            module._gguf_type = entry[0]
            model._q8_oracle_hooks.append(module.register_forward_pre_hook(pack_input))
    quantized = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            native = name_map(name)
            if native in stored:
                typ, data = stored[native]
                parameter.copy_(torch.from_numpy(data.copy()).reshape(parameter.shape))
                quantized += typ == 8
    assert quantized > 0, 'fixture did not exercise any quantized tensors'
    return quantized
