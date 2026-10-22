"""Shared safe checkpoint loading and GGUF serialization."""
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


def get_model_architecture(source, clamp3_config=None, muq_revision=None, muq_mulan_revision=None):
    """Detect the architecture using the existing checkpoint selection rules."""
    config_path = source / 'config.json'
    hparams = json.loads(config_path.read_text()) if source.is_dir() and config_path.exists() else {}
    if muq_revision is not None or ('encoder_dim' in hparams and 'w2v2_config' in hparams):
        return 'muq'
    if hparams.get('model_type') == 'mert2':
        return 'mert2'
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
