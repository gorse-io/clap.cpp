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

from importlib import import_module

from .base import ModelBase, get_model_architecture


__all__ = ['ModelBase', 'get_model_architecture', 'get_model_class']


MODEL_MAP = {
    'clap': 'clap',
    'clap2': 'clap2',
    'clamp3': 'clamp3',
    'mert': 'mert',
    'mert2': 'mert2',
    'muq': 'muq',
    'muq_mulan': 'muq_mulan',
}


def get_model_class(model_type):
    """Import and return the converter registered for this architecture."""
    if model_type not in MODEL_MAP:
        raise NotImplementedError(f'Architecture {model_type!r} not supported!')
    import_module(f'.{MODEL_MAP[model_type]}', __name__)
    return ModelBase._model_classes[model_type]
