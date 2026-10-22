"""MuQ non-dense quantized storage is unsupported; never crash or publish."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

native, quantizer = map(Path, sys.argv[1:])


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


with tempfile.TemporaryDirectory(prefix='MuQ quantized ') as temporary:
    output = Path(temporary)
    for name, kind, size in [('Q8_0', 8, 272), ('Q4_K', 12, 144), ('Q6_K', 14, 210)]:
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
        header += string('general.architecture') + struct.pack('<I', 8) + string('muq')
        header += string('quantized.weight') + struct.pack('<IQIQ', 1, 256, kind, 0)
        source = output / (name + '.gguf')
        source.write_bytes(header + bytes(-len(header) % 32) + bytes(size))
        result = subprocess.run([str(native), '--model', str(source), '--text', 'unused'],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode > 0 and 'quantized non-dense/embedding tensor' in result.stderr, (name, result)
        destination = output / (name + '-converted.gguf')
        result = subprocess.run([str(quantizer), 'quantize', str(source), str(destination)],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode > 0 and not destination.exists(), (name, result)
    assert not list(output.glob('*.tmp-*'))
print('MuQ non-dense Q8_0/Q4_K/Q6_K rejection passed')
