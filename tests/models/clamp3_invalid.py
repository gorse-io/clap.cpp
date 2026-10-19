#!/usr/bin/env python3
"""Reject malformed architecture metadata before graph execution."""
import json
from pathlib import Path
import struct
import subprocess
import sys

binary, fixture = Path(sys.argv[1]), Path(sys.argv[2])
source = (fixture / 'model.gguf').read_bytes()
_, version, tensors, count = struct.unpack_from('<4sIQQ', source)
assert version == 3
position = 24

def read_string():
    global position
    size, = struct.unpack_from('<Q', source, position)
    position += 8
    value = source[position:position + size].decode()
    position += size
    return value

metadata = []
for _ in range(count):
    key = read_string()
    kind, = struct.unpack_from('<I', source, position)
    position += 4
    assert kind == 8
    metadata.append((key, read_string()))
infos_start = position
for _ in range(tensors):
    read_string()
    dimensions, = struct.unpack_from('<I', source, position)
    position += 4 + dimensions * 8 + 12
infos = source[infos_start:position]
payload = source[(position + 31) // 32 * 32:]

def encode_string(text):
    value = text.encode()
    return struct.pack('<Q', len(value)) + value

cases = [('negative-pad', 'text_config', 'pad_token_id', -1, 'padding token'),
         ('negative-epsilon', 'audio_config', 'layer_norm_eps', -1, 'layer norm epsilon'),
         ('causal-decoder', 'text_config', 'is_decoder', True, 'decoder')]
for name, branch, key, value, expected in cases:
    header = bytearray(source[:24])
    for field, original in metadata:
        if field == 'clamp3.config':
            config = json.loads(original)
            config[branch][key] = value
            original = json.dumps(config)
        header += encode_string(field) + struct.pack('<I', 8) + encode_string(original)
    header += infos
    header += b'\0' * (-len(header) % 32)
    path = fixture / (name + '.gguf')
    path.write_bytes(header + payload)
    result = subprocess.run([str(binary), '--model', str(path), '--text', 'ab'], capture_output=True, text=True)
    assert result.returncode != 0 and expected in result.stderr, (name, result.returncode, result.stderr)
print('Malformed CLaMP3 metadata rejected:', len(cases), 'cases')
