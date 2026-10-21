"""Discover model-specific Hugging Face to GGUF converters on demand."""
from importlib import import_module

from .base import ModelBase, get_model_architecture


__all__ = ['ModelBase', 'get_model_architecture', 'get_model_class']


MODEL_MAP = {
    'clap': 'clap',
    'clamp3': 'clamp3',
    'mert2': 'mert2',
}


def get_model_class(model_type):
    """Import and return the converter registered for this architecture."""
    if model_type not in MODEL_MAP:
        raise NotImplementedError(f'Architecture {model_type!r} not supported!')
    import_module(f'.{MODEL_MAP[model_type]}', __name__)
    return ModelBase._model_classes[model_type]
