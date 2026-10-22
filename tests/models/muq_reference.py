"""Pinned official MuQ inference components; no training/RVQ dependencies.

The wiring is MuQModel.preprocessing/normalize/encoder and MuQ.forward from
Tencent's MuQ revision below. Quantizers, classifier, CLS and positional conv
are unused by that inference path. Never execute unverified custom source.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
from transformers import Wav2Vec2ConformerConfig
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerEncoder

SOURCE_REVISION = '28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a'
MODEL_REVISION = '0562a57814f6f8bbd9fdea0a25921a2fce1a841a'
SOURCE_HASHES = {
    'conv.py': '98386c207de5e70ff02d3cb76eac276902556faf4cd9902a5f8b7b2f0e27b221',
    'features.py': 'd4bd1b49bce611bebb8133a4bd4dbfd6252bc7554d96a0b13119be190b62945e',
}


def official_module(directory, name):
    path = Path(directory) / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_HASHES[name]:
        raise ValueError(f'unverified official MuQ source: {name}: {digest}')
    spec = importlib.util.spec_from_file_location('official_muq_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InferenceCore(nn.Module):
    def __init__(self, raw, reference):
        super().__init__()
        conv = official_module(reference, 'conv.py')
        features = official_module(reference, 'features.py')
        self.raw = raw
        with torch.device('cpu'):
            self.frontend = features.MelSTFT(n_fft=2048, hop_length=raw['hop_length'], is_db=True)
        self.conv = conv.Conv2dSubsampling(1, raw['conv_dim'], raw['encoder_dim'],
                                          strides=[2, 2], n_bands=128)
        config = Wav2Vec2ConformerConfig(**raw['w2v2_config'])
        config.hidden_size = raw['encoder_dim']
        config.num_hidden_layers = raw['encoder_depth']
        self.conformer = Wav2Vec2ConformerEncoder(config)

    def forward(self, waveform):
        features = self.frontend(waveform.float())[..., :-1]
        features = (features - self.raw['stat']['melspec_2048_mean']) / self.raw['stat']['melspec_2048_std']
        hidden = self.conv(features)
        return self.conformer(hidden, output_hidden_states=True).hidden_states[-1]


def inference_key(name):
    return (name.startswith(('model.conv.', 'model.conformer.layers.', 'model.conformer.layer_norm.'))
            and not name.endswith('.num_batches_tracked'))
