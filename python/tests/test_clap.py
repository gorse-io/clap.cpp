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
import math
from pathlib import Path

import pytest

from huggingface_hub import hf_hub_download

from clap_cpp import CLAP


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def test_image_api():
    assert callable(getattr(CLAP, "encode_image", None))


def test_clap():
    model_path = hf_hub_download(
        repo_id="gorse-io/larger_clap_general-GGUF",
        filename="larger_clap_general-q4_k_m.gguf",
    )

    audio_path = Path(__file__).resolve().parents[2] / "assets" / "dog.wav"
    with CLAP(model_path) as clap:
        with pytest.raises(ValueError, match="NUL"):
            clap.encode_image("bad\0.png")
        with pytest.raises(TypeError):
            clap.encode_image(b"image.png")
        # Unsupported images must not invalidate an audio/text-only model.
        with pytest.raises(RuntimeError):
            clap.encode_image(audio_path.parent / "clap.png")
        audio_embedding = clap.encode_audio(audio_path, 0)
        text_embedding_1 = clap.encode_text("Sound of a dog")
        text_embedding_2 = clap.encode_text("Sound of vaccum cleaner")

    with pytest.raises(RuntimeError, match="closed"):
        clap.encode_image(audio_path.parent / "clap.png")

    distance_1 = cosine(audio_embedding, text_embedding_1)
    distance_2 = cosine(audio_embedding, text_embedding_2)
    # softmax(cosine * exp(logit_scale_a)), as in Transformers ClapModel.
    # logit_scale_a is the learned scalar from laion/larger_clap_general's
    # pytorch_model.bin, not the configuration's initialization value.
    logit_scale = math.exp(3.6549275)
    logit_1 = distance_1 * logit_scale
    logit_2 = distance_2 * logit_scale
    max_logit = max(logit_1, logit_2)
    exp_1 = math.exp(logit_1 - max_logit)
    exp_2 = math.exp(logit_2 - max_logit)
    score_1 = exp_1 / (exp_1 + exp_2)
    score_2 = exp_2 / (exp_1 + exp_2)
    assert score_1 > 0.99
    assert score_2 < 0.01
