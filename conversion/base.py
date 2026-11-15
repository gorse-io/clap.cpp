# Copyright 2026 clap.cpp Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import struct
import tempfile


def string(value):
    value = value.encode('utf-8')
    return struct.pack('<Q', len(value)) + value


def load_checkpoint(path, torch):
    try:
        weights = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except TypeError as error:
        raise RuntimeError('torch with safe mmap checkpoint loading is required') from error
    if isinstance(weights, dict) and isinstance(weights.get('model'), dict):
        weights = weights['model']
    if not isinstance(weights, dict):
        raise ValueError('checkpoint does not contain a state dictionary')
    return weights


def write_gguf(destination, tensors, metadata, dtype, np, torch, keep_f32=None):
    encoded = []
    for name, value in tensors:
        if len(name.encode()) >= 64:
            raise ValueError(f'tensor name exceeds ggml limit: {name}')
        half = dtype == 'f16' and value.ndim > 1 and not (keep_f32 and keep_f32(name, value))
        encoded.append((name, value, np.dtype('<f2' if half else '<f4'), 1 if half else 0))
    header = b'GGUF' + struct.pack('<IQQ', 3, len(encoded), len(metadata))
    for key, value in metadata.items():
        header += string(key) + struct.pack('<I', 8) + string(value)
    offset = 0
    for name, value, dt, typ in encoded:
        header += string(name) + struct.pack('<I', value.ndim)
        header += struct.pack('<' + 'Q' * value.ndim, *reversed(value.shape))
        header += struct.pack('<IQ', typ, offset)
        offset += (value.numel() * dt.itemsize + 31) // 32 * 32
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(header)
            output.write(b'\0' * ((-len(header)) % 32))
            for _, value, dt, typ in encoded:
                data = value.detach().to(dtype=torch.float16 if typ else torch.float32).contiguous().numpy()
                data.astype(dt, copy=False).tofile(output)
                output.write(b'\0' * ((-data.nbytes) % 32))
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print(f'{destination}: {len(encoded)} tensors ({dtype}, {metadata["general.architecture"]})')


def write_gguf_streaming(destination, tensors, metadata, dtype):
    """Atomically serialize file-backed float tensors with bounded working memory.

    Descriptors contain name, HF-order shape, dtype (F32/F16/BF16), path and
    absolute byte offset. Unlike write_gguf, no checkpoint tensors are retained.
    """
    import array
    import math
    import sys

    if dtype not in ('f32', 'f16'):
        raise ValueError('unsupported GGUF storage dtype')
    encoded, names, offset = [], set(), 0
    for tensor in tensors:
        name, shape = tensor['name'], tensor['shape']
        if not name or len(name.encode('utf-8')) >= 64 or name in names:
            raise ValueError(f'invalid or duplicate tensor name: {name}')
        names.add(name)
        if not 1 <= len(shape) <= 4 or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError(f'invalid tensor shape: {name}')
        if tensor['dtype'] not in ('F32', 'F16', 'BF16'):
            raise ValueError(f'unsupported tensor dtype: {name}')
        if type(tensor['offset']) is not int or tensor['offset'] < 0:
            raise ValueError(f'invalid tensor offset: {name}')
        half = dtype == 'f16' and len(shape) > 1
        size = math.prod(shape) * (2 if half else 4)
        encoded.append((tensor, 1 if half else 0, offset, size))
        offset += (size + 31) // 32 * 32
    header = b'GGUF' + struct.pack('<IQQ', 3, len(encoded), len(metadata))
    for key, value in metadata.items():
        header += string(key) + struct.pack('<I', 8) + string(value)
    for tensor, typ, offset, _ in encoded:
        shape = tensor['shape']
        header += string(tensor['name']) + struct.pack('<I', len(shape))
        header += struct.pack('<' + 'Q' * len(shape), *reversed(shape))
        header += struct.pack('<IQ', typ, offset)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(header)
            output.write(b'\0' * (-len(header) % 32))
            for tensor, typ, _, size in encoded:
                remaining = math.prod(tensor['shape'])
                source_dtype = tensor['dtype']
                width = 4 if source_dtype == 'F32' else 2
                with open(tensor['path'], 'rb') as source:
                    source.seek(tensor['offset'])
                    while remaining:
                        count = min(remaining, 65536)
                        block = source.read(count * width)
                        if len(block) != count * width:
                            raise ValueError(f'truncated tensor: {tensor["name"]}')
                        if (source_dtype == 'F32' and typ == 0 or
                                source_dtype == 'F16' and typ == 1):
                            output.write(block)
                        else:
                            if source_dtype == 'BF16':
                                words = array.array('H')
                                words.frombytes(block)
                                if sys.byteorder != 'little':
                                    words.byteswap()
                                floats = array.array('I', (word << 16 for word in words))
                                if sys.byteorder != 'little':
                                    floats.byteswap()
                                block = floats.tobytes()
                                values = struct.unpack('<' + str(count) + 'f', block) if typ else ()
                            else:
                                values = struct.unpack('<' + str(count) + ('f' if width == 4 else 'e'), block)
                            if typ:
                                # IEEE half overflow saturates to infinity, as torch/numpy do.
                                values = [math.copysign(math.inf, v) if abs(v) >= 65520 else v for v in values]
                                output.write(struct.pack('<' + str(count) + 'e', *values))
                            elif source_dtype == 'BF16':
                                output.write(block)
                            else:
                                output.write(struct.pack('<' + str(count) + 'f', *values))
                        remaining -= count
                output.write(b'\0' * (-size % 32))
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'{destination}: {len(encoded)} tensors ({dtype}, {metadata["general.architecture"]})')


def get_model_architecture(source, clamp3_config=None, muq_revision=None, muq_mulan_revision=None):
    """Detect the architecture using the existing checkpoint selection rules."""
    config_path = source / 'config.json'
    hparams = json.loads(config_path.read_text()) if source.is_dir() and config_path.exists() else {}
    if muq_revision is not None or ('encoder_dim' in hparams and 'w2v2_config' in hparams):
        return 'muq'
    if hparams.get('model_type') == 'mert2':
        return 'mert2'
    if hparams.get('model_type') == 'clap2':
        return 'clap2'
    if hparams.get('model_type') == 'qwen2_5_omni_thinker':
        return 'voiceclap'
    if hparams.get('model_type') == 'mert_model':
        return 'mert'
    if muq_mulan_revision is not None or all(key in hparams for key in ('mulan', 'audio_model', 'text_model')):
        return 'muq_mulan'
    if clamp3_config is not None or source.is_file():
        return 'clamp3'
    return 'clap'


class ModelBase(ABC):
    """Common converter options and architecture registration."""

    _model_classes = {}

    def __init__(self, source, destination, dtype='f32', **options):
        self.source = Path(source)
        self.destination = Path(destination)
        self.dtype = dtype
        self.options = options

    @classmethod
    def register(cls, *architectures):
        def decorator(model_class):
            for architecture in architectures:
                cls._model_classes[architecture] = model_class
            return model_class
        return decorator

    @abstractmethod
    def write(self):
        """Write the model's tensors and metadata to GGUF."""
