"""Small deterministic, valid CLAP GGUF using only the Python standard library.

All encoder weights are zero; a nonzero output bias makes the normalized
projection deterministic. This exercises real model loading/tokenization/inference,
not numerical parity (covered by the existing optional model oracle tests).
"""
import json
import math
import struct


def tiny_clap():
    audio = dict(enable_fusion=False, depths=[1]*4, num_attention_heads=[1]*4,
                 patch_size=4, patch_stride=[4, 4], patch_embed_input_channels=1,
                 qkv_bias=True, enable_patch_layer_norm=True, flatten_patch_embeds=True,
                 hidden_act="gelu", projection_hidden_act="relu", spec_size=32,
                 num_mel_bins=32, window_size=1, patch_embeds_hidden_size=2,
                 hidden_size=16, mlp_ratio=2, projection_dim=2)
    text = dict(hidden_act="gelu", projection_hidden_act="relu", hidden_size=4,
                num_attention_heads=1, num_hidden_layers=1, type_vocab_size=1,
                vocab_size=7, max_position_embeddings=32, intermediate_size=8,
                projection_dim=2, bos_token_id=0, eos_token_id=2, pad_token_id=1,
                layer_norm_eps=1e-5)
    config = dict(model_type="clap", audio_config=audio, text_config=text, projection_dim=2)
    tokenizer = dict(model=dict(type="BPE", vocab={"<s>": 0, "<pad>": 1, "</s>": 2,
                                                 "h": 3, "e": 4, "l": 5, "o": 6}, merges=[]))
    metadata = {"general.architecture": "clap", "clap.config": json.dumps(config),
                "clap.tokenizer": json.dumps(tokenizer)}
    tensors = []

    def tensor(name, shape):
        # A positive projection bias avoids a zero-norm output with zero weights.
        value = 1.0 if name == "text_proj.linear2.bias" else 0.0
        tensors.append((name, shape, struct.pack("<f", value) * math.prod(shape)))

    def norm(name, dim):
        tensor(name + ".weight", [dim])
        tensor(name + ".bias", [dim])

    def linear(name, src, dst, bias=True):
        tensor(name + ".weight", [src, dst])
        if bias:
            tensor(name + ".bias", [dst])

    tensor("audio.patch_embed.proj.weight", [4, 4, 1, 2])
    tensor("audio.patch_embed.proj.bias", [2])
    norm("audio.patch_embed.norm", 2)
    norm("audio.batch_norm", 32)
    tensor("audio.batch_norm.running_mean", [32])
    tensor("audio.batch_norm.running_var", [32])
    for stage in range(4):
        dim = 2 << stage
        p = f"audio.layers.{stage}.blocks.0"
        norm(p + ".layernorm_before", dim)
        norm(p + ".layernorm_after", dim)
        for q in ("query", "key", "value"):
            linear(p + ".attention.self." + q, dim, dim)
        tensor(p + ".attention.self.rel_bias", [1, 1])
        linear(p + ".attention.output.dense", dim, dim)
        linear(p + ".intermediate.dense", dim, dim * 2)
        linear(p + ".output.dense", dim * 2, dim)
        if stage < 3:
            p = f"audio.layers.{stage}.downsample"
            norm(p + ".norm", dim * 4)
            linear(p + ".reduction", dim * 4, dim * 2, False)
    norm("audio.norm", 16)
    for name, size in (("word", 7), ("position", 32), ("token_type", 1)):
        tensor(f"text.embeddings.{name}_embeddings.weight", [4, size])
    norm("text.embeddings.LayerNorm", 4)
    p = "text.encoder.layer.0"
    for q in ("query", "key", "value"):
        linear(p + ".attention.self." + q, 4, 4)
    linear(p + ".attention.output.dense", 4, 4)
    norm(p + ".attention.output.LayerNorm", 4)
    linear(p + ".intermediate.dense", 4, 8)
    linear(p + ".output.dense", 8, 4)
    norm(p + ".output.LayerNorm", 4)
    linear("text.pooler.dense", 4, 4)
    for branch, dim in (("audio", 16), ("text", 4)):
        linear(branch + "_proj.linear1", dim, 2)
        linear(branch + "_proj.linear2", 2, 2)

    def string(s):
        data = s.encode()
        return struct.pack("<Q", len(data)) + data

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += string(key) + struct.pack("<I", 8) + string(value)
    body = b""
    for name, shape, data in tensors:
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", 0, len(body))
        body += data
        body += b"\0" * (-len(body) % 32)
    return header + b"\0" * (-len(header) % 32) + body
