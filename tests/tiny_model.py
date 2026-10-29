"""Stdlib-only tiny, valid CLAP GGUF and independent scalar text oracle.

write_model(path, *, broken_projection=False, waveform=False) returns metadata
with dimensions, model (filename), and cases [{input, tokens, embedding}].
No downloads or torch. The default reduced-mel model is text/tensor-only;
waveform=True enables the 64-bin, 256-spec waveform frontend configuration.
"""

import json
import math
from pathlib import Path
import struct


def write_model(path, *, broken_projection=False, waveform=False) -> dict:
    path = Path(path)
    text = dict(hidden_size=8, intermediate_size=16, num_attention_heads=2,
                num_hidden_layers=1, vocab_size=261, type_vocab_size=1,
                max_position_embeddings=514, bos_token_id=0, pad_token_id=1,
                eos_token_id=2, hidden_act="gelu", projection_hidden_act="relu",
                projection_dim=5, layer_norm_eps=1e-5)
    audio = dict(enable_fusion=False, depths=[1] * 4, num_attention_heads=[1, 1, 2, 4],
                 patch_size=4, patch_stride=[4, 4], patch_embed_input_channels=1,
                 qkv_bias=True, enable_patch_layer_norm=True, flatten_patch_embeds=True,
                 hidden_act="gelu", projection_hidden_act="relu", spec_size=32,
                 num_mel_bins=32, window_size=1, patch_embeds_hidden_size=2,
                 hidden_size=16, mlp_ratio=2.0, projection_dim=5)
    if waveform:
        audio.update(spec_size=256, num_mel_bins=64, layer_norm_eps=1e-5)
    config = dict(model_type="clap", projection_dim=5, text_config=text, audio_config=audio)
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
    extra = 256
    for byte in range(256):
        regular = 33 <= byte <= 126 or 161 <= byte <= 172 or byte >= 174
        vocab[chr(byte if regular else extra)] = byte + 5
        if not regular:
            extra += 1
    tokenizer = dict(model=dict(type="BPE", vocab=vocab, merges=[]))
    tensors = {}

    def add(name, shape, norm=False):
        serial = len(tensors) + 1
        count = math.prod(shape)
        values = [1.0 if norm else 0.18 * math.sin((i + serial * 17) * 0.173)
                  for i in range(count)]
        if broken_projection and name.startswith("text_proj.linear2."):
            values = [0.0] * count
        # The oracle uses exactly the F32 storage values, not Python doubles.
        data = struct.pack("<" + "f" * count, *values)
        tensors[name] = (shape, list(struct.unpack("<" + "f" * count, data)), data)

    def norm(name, dim):
        add(name + ".weight", [dim], True)
        add(name + ".bias", [dim])

    def linear(name, inputs, outputs, bias=True):
        add(name + ".weight", [inputs, outputs])
        if bias:
            add(name + ".bias", [outputs])

    add("audio.patch_embed.proj.weight", [4, 4, 1, 2])
    add("audio.patch_embed.proj.bias", [2])
    norm("audio.patch_embed.norm", 2)
    norm("audio.batch_norm", audio["num_mel_bins"])
    add("audio.batch_norm.running_mean", [audio["num_mel_bins"]])
    add("audio.batch_norm.running_var", [audio["num_mel_bins"]], True)
    dim = 2
    for stage in range(4):
        p = f"audio.layers.{stage}.blocks.0"
        norm(p + ".layernorm_before", dim)
        norm(p + ".layernorm_after", dim)
        for q in ("query", "key", "value"):
            linear(p + ".attention.self." + q, dim, dim)
        add(p + ".attention.self.rel_bias", [[1, 1, 2, 4][stage], 1])
        linear(p + ".attention.output.dense", dim, dim)
        linear(p + ".intermediate.dense", dim, dim * 2)
        linear(p + ".output.dense", dim * 2, dim)
        if stage < 3:
            down = f"audio.layers.{stage}.downsample"
            norm(down + ".norm", dim * 4)
            linear(down + ".reduction", dim * 4, dim * 2, False)
        dim *= 2
    norm("audio.norm", 16)
    add("text.embeddings.word_embeddings.weight", [8, 261])
    add("text.embeddings.position_embeddings.weight", [8, 514])
    add("text.embeddings.token_type_embeddings.weight", [8, 1])
    norm("text.embeddings.LayerNorm", 8)
    p = "text.encoder.layer.0"
    for q in ("query", "key", "value"):
        linear(p + ".attention.self." + q, 8, 8)
    linear(p + ".attention.output.dense", 8, 8)
    norm(p + ".attention.output.LayerNorm", 8)
    linear(p + ".intermediate.dense", 8, 16)
    linear(p + ".output.dense", 16, 8)
    norm(p + ".output.LayerNorm", 8)
    linear("text.pooler.dense", 8, 8)
    linear("text_proj.linear1", 8, 5)
    linear("text_proj.linear2", 5, 5)
    linear("audio_proj.linear1", 16, 5)
    linear("audio_proj.linear2", 5, 5)

    def string(value):
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    metadata = {"general.architecture": "clap", "clap.config": json.dumps(config),
                "clap.tokenizer": json.dumps(tokenizer)}
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for name, value in metadata.items():
        header += string(name) + struct.pack("<I", 8) + string(value)
    data = bytearray()
    for name, (shape, _, raw) in tensors.items():
        header += (string(name) + struct.pack("<I", len(shape))
                   + struct.pack("<" + "Q" * len(shape), *shape)
                   + struct.pack("<IQ", 0, len(data)))
        data.extend(raw)
        data.extend(b"\0" * (-len(data) % 32))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"\0" * (-len(header) % 32) + data)

    def weight(name):
        return tensors[name][1]

    def dense(rows, name):
        shape, w, _ = tensors[name + ".weight"]
        width, outputs = shape
        bias = weight(name + ".bias")
        return [[sum(row[i] * w[j * width + i] for i in range(width)) + bias[j]
                 for j in range(outputs)] for row in rows]

    def normalize(rows, name):
        w, b = weight(name + ".weight"), weight(name + ".bias")
        result = []
        for row in rows:
            mean = sum(row) / len(row)
            scale = 1 / math.sqrt(sum((v - mean) ** 2 for v in row) / len(row) + 1e-5)
            result.append([(v - mean) * scale * w[i] + b[i] for i, v in enumerate(row)])
        return result

    def residual(left, right):
        return [[a + b for a, b in zip(x, y)] for x, y in zip(left, right)]

    def reference(prompt):
        ids = [0] + [byte + 5 for byte in prompt.encode("utf-8")] + [2]
        ids = ids[:511] + [2] if len(ids) > 512 else ids
        x = []
        for pos, token in enumerate(ids, 2):
            x.append([weight("text.embeddings.word_embeddings.weight")[token * 8 + i]
                      + weight("text.embeddings.position_embeddings.weight")[pos * 8 + i]
                      + weight("text.embeddings.token_type_embeddings.weight")[i] for i in range(8)])
        x = normalize(x, "text.embeddings.LayerNorm")
        q, k, v = [dense(x, p + ".attention.self." + part) for part in ("query", "key", "value")]
        attention = [[0.0] * 8 for _ in ids]
        for h in range(2):
            for i in range(len(ids)):
                scores = [sum(q[i][h * 4 + c] * k[j][h * 4 + c] for c in range(4)) / 2
                          for j in range(len(ids))]
                maximum = max(scores)
                scores = [math.exp(s - maximum) for s in scores]
                denominator = sum(scores)
                for c in range(4):
                    attention[i][h * 4 + c] = sum(s / denominator * v[j][h * 4 + c]
                                                 for j, s in enumerate(scores))
        x = normalize(residual(dense(attention, p + ".attention.output.dense"), x),
                      p + ".attention.output.LayerNorm")
        y = dense(x, p + ".intermediate.dense")
        y = [[0.5 * value * (1 + math.erf(value / math.sqrt(2))) for value in row] for row in y]
        x = normalize(residual(dense(y, p + ".output.dense"), x), p + ".output.LayerNorm")
        pooled = [[math.tanh(value) for value in dense([x[0]], "text.pooler.dense")[0]]]
        projected = dense([[max(value, 0) for value in dense(pooled, "text_proj.linear1")[0]]],
                          "text_proj.linear2")[0]
        length = math.sqrt(sum(value * value for value in projected))
        return dict(input=prompt, tokens=len(ids),
                    embedding=[value / length for value in projected] if length else [])

    return dict(model=path.name, dimensions=5,
                cases=[reference(prompt) for prompt in ("hello", "world", "hello world!", "caf\u00e9", "a", "b")])
