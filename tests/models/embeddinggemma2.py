"""Generated tiny GGUF + independent stdlib scalar EmbeddingGemma2 oracle.

Contract: llama.cpp src/models/gemma-embedding2.cpp, GGUF architecture
'gemma-embedding2'. No downloads, optional ML packages, or committed binaries.
Run: python embeddinggemma2.py --test /path/to/clap-embeddinggemma2-test
Generate: python embeddinggemma2.py --output /path/to/fixtures
"""
import argparse
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile

ARCH = "gemma-embedding2"


def string(s):
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def typed(value):
    if isinstance(value, str):
        return 8, string(value)
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        return 5, struct.pack("<i", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    element, _ = typed(value[0])
    return 9, struct.pack("<IQ", element, len(value)) + b"".join(typed(v)[1] for v in value)


def save(path, metadata, tensors, storage):
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        kind, data = typed(value)
        header += string(key) + struct.pack("<I", kind) + data
    blob = bytearray()
    for name, (shape, values) in tensors.items():
        dtype = 1 if storage == "f16" and len(shape) == 2 else 0
        raw = struct.pack("<" + ("e" if dtype else "f") * len(values), *values)
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", dtype, len(blob))
        blob.extend(raw)
        blob.extend(b"\0" * (-len(blob) % 32))
    path.write_bytes(header + b"\0" * (-len(header) % 32) + blob)


def fixture(storage="f32", array_pattern=False, audio_soft=None):
    hidden, layers, ple, ff, vocab, context, output = 8, 3, 3, 13, 271, 16, 5
    if audio_soft is not None:
        vocab += 2
    heads, kv = [4, 4, 4], [2, 1, 2]
    # First/third layers local, second global: unequal head sizes and partial
    # global RoPE deliberately exercise more than the real model's geometry.
    dims, rotary, local = [4, 8, 4], [4, 4, 4], [True, False, True]
    metadata: dict = {"general.architecture": ARCH, "general.name": "tiny-generated-embeddinggemma2"}
    fields = dict(context_length=context, embedding_length=hidden, embedding_length_out=output,
                  embedding_length_per_layer_input=ple, block_count=layers, feed_forward_length=[ff]*layers if array_pattern else ff,
                  pooling_type=1)
    for key, value in fields.items():
        metadata[ARCH + "." + key] = value
    fields = {"head_count": heads if array_pattern else 4, "head_count_kv": kv, "key_length": 8, "value_length": 8,
              "key_length_swa": 4, "value_length_swa": 4, "layer_norm_rms_epsilon": 1e-5,
              "sliding_window": 4, "sliding_window_pattern": [True, False, True] if array_pattern else 2,
              "causal": False}
    for key, value in fields.items():
        metadata[ARCH + ".attention." + key] = value
    metadata.update({ARCH + ".rope.freq_base": 10000.0, ARCH + ".rope.freq_base_swa": 1000.0,
                     ARCH + ".rope.dimension_count": 4, ARCH + ".rope.dimension_count_swa": 4,
                     "tokenizer.ggml.model": "gemma4", "tokenizer.ggml.pre": "default",
                     "tokenizer.ggml.tokens": ["<pad>", "<eos>", "<bos>", "<unk>", "▁a", "▁b", "c", "d", "e", "f", "g", "h", "▁", "a", "b"] + [f"<0x{i:02X}>" for i in range(256)],
                     "tokenizer.ggml.scores": [0.0] * vocab,
                     "tokenizer.ggml.token_type": [3, 3, 3, 2] + [1] * 11 + [6] * 256,
                     "tokenizer.ggml.merges": ["▁ a", "▁ b"],
                     "tokenizer.ggml.bos_token_id": 2, "tokenizer.ggml.eos_token_id": 1,
                     "tokenizer.ggml.padding_token_id": 0, "tokenizer.ggml.unknown_token_id": 3,
                     "tokenizer.ggml.add_bos_token": True, "tokenizer.ggml.add_eos_token": True})
    if audio_soft is not None:
        metadata["tokenizer.ggml.tokens"] += ["<|audio>", "<audio|>"]
        metadata["tokenizer.ggml.token_type"] += [3, 3]
    tensors = {}

    def add(name, shape, norm=False, scale=False):
        serial = len(tensors) + 1
        values = [(0.9 + 0.17 * math.sin(i + serial) if norm else
                   0.83 + 0.03 * serial if scale else
                   0.23 * math.sin((i + serial * 11) * 0.311) + 0.04 * math.cos(i * 0.13))
                  for i in range(math.prod(shape))]
        fmt = "e" if storage == "f16" and len(shape) == 2 else "f"
        values = list(struct.unpack("<" + fmt * len(values), struct.pack("<" + fmt * len(values), *values)))
        tensors[name + ".weight"] = (shape, values)

    add("token_embd", [hidden, vocab])
    add("per_layer_model_proj", [hidden, ple * layers])
    add("per_layer_proj_norm", [ple], norm=True)
    add("output_norm", [hidden], norm=True)
    add("output", [hidden, output])
    for layer, dim in enumerate(dims):
        root = f"blk.{layer}."
        for name, size in [("attn_norm", hidden), ("attn_q_norm", dim), ("attn_k_norm", dim),
                           ("post_attention_norm", hidden), ("ffn_norm", hidden),
                           ("post_ffw_norm", hidden), ("post_norm", hidden)]:
            add(root + name, [size], norm=True)
        for name, ins, outs in [("attn_q", hidden, heads[layer] * dim), ("attn_k", hidden, kv[layer] * dim),
                                ("attn_v", hidden, kv[layer] * dim), ("attn_output", heads[layer] * dim, hidden),
                                ("ffn_gate", hidden, ff), ("ffn_up", hidden, ff), ("ffn_down", ff, hidden),
                                ("inp_gate", hidden, ple), ("proj", ple, hidden)]:
            add(root + name, [ins, outs])
        add(root + "layer_output_scale", [1], scale=True)

    def weight(name):
        return tensors[name + ".weight"][1]

    def dense(rows, name):
        (width, outs), w = tensors[name + ".weight"]
        # Weights are storage-rounded above; inference preserves F32 operands
        # even for F16 storage (EmbeddingGemma2 can overflow F16 activations).
        return [[sum(row[i] * w[j * width + i] for i in range(width))
                 for j in range(outs)] for row in rows]

    def norm(rows, name=None):
        w = weight(name) if name else None
        return [[v / math.sqrt(sum(x*x for x in row) / len(row) + 1e-5) * (w[i] if w else 1)
                 for i, v in enumerate(row)] for row in rows]

    def plus(a, b):
        return [[x+y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

    def gelu(x):
        return 0.5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*x*x*x)))

    def reference(ids, initial=None):
        x = [list(row) for row in initial] if initial is not None else [[weight("token_embd")[token*hidden+i] * math.sqrt(hidden) for i in range(hidden)] for token in ids]
        all_ple = [[v / math.sqrt(hidden) for v in row] for row in dense(x, "per_layer_model_proj")]
        inputs = [norm([row[l*ple:(l+1)*ple] for row in all_ple], "per_layer_proj_norm") for l in range(layers)]
        for layer, dim in enumerate(dims):
            root = f"blk.{layer}."
            pre = norm(x, root + "attn_norm")
            q, k, v = [dense(pre, root + "attn_" + name) for name in ["q", "k", "v"]]
            def head_norm(rows, count, name):
                return [sum(norm([row[h*dim:(h+1)*dim] for h in range(count)], name), []) for row in rows]
            q = head_norm(q, heads[layer], root + "attn_q_norm")
            k = head_norm(k, kv[layer], root + "attn_k_norm")
            v = head_norm(v, kv[layer], None)
            for rows, count in [(q, heads[layer]), (k, kv[layer])]:
                for pos, row in enumerate(rows):
                    for head in range(count):
                        # GGUF Gemma uses NEOX pairs in the first n_rot channels.
                        for i in range(rotary[layer] // 2):
                            angle = pos * (1000.0 if local[layer] else 10000.0) ** (-2*i/rotary[layer])
                            a, b = head*dim+i, head*dim+i+rotary[layer]//2
                            row[a], row[b] = row[a]*math.cos(angle)-row[b]*math.sin(angle), row[a]*math.sin(angle)+row[b]*math.cos(angle)
            attended = [[0.0]*(heads[layer]*dim) for _ in ids]
            for pos in range(len(ids)):
                for head in range(heads[layer]):
                    kh = head // (heads[layer] // kv[layer])
                    positions = [j for j in range(len(ids)) if ids[j] != 0 and (not local[layer] or abs(pos-j) <= 4 // 2)]
                    if not positions:
                        continue
                    scores = [sum(q[pos][head*dim+c]*k[j][kh*dim+c] for c in range(dim)) for j in positions]
                    scores = [math.exp(s-max(scores)) for s in scores]
                    for c in range(dim):
                        attended[pos][head*dim+c] = sum(s*v[j][kh*dim+c] for j, s in zip(positions, scores))/sum(scores)
            x = plus(x, norm(dense(attended, root + "attn_output"), root + "post_attention_norm"))
            pre = norm(x, root + "ffn_norm")
            gate, up = dense(pre, root + "ffn_gate"), dense(pre, root + "ffn_up")
            gated = [[gelu(a)*b for a, b in zip(ra, rb)] for ra, rb in zip(gate, up)]
            x = plus(x, norm(dense(gated, root + "ffn_down"), root + "post_ffw_norm"))
            gated = [[gelu(a)*b for a, b in zip(ra, rb)] for ra, rb in zip(dense(x, root + "inp_gate"), inputs[layer])]
            x = plus(x, norm(dense(gated, root + "proj"), root + "post_norm"))
            x = [[v*weight(root+"layer_output_scale")[0] for v in row] for row in x]
        # Project each token before pooling, as in the reference graph. Moving
        # mean ahead of a quantized/F16 matmul changes input packing rounding.
        x = dense(norm(x, "output_norm"), "output")
        pooled = [sum(row[c] for token, row in zip(ids, x) if token != 0)/sum(token != 0 for token in ids) for c in range(output)]
        length = math.sqrt(sum(v*v for v in pooled))
        return [v/length for v in pooled]

    cases = [[2, 4, 1], [2, 5, 6, 7, 8, 9, 1], [4], [2, 4, 5, 4, 1], [2] + [4, 5]*7 + [1],
             [2, 4, 1, 0, 0, 0, 0], [0, 0, 2, 4, 1], [2, 4, 0, 5, 0, 1]]
    # Storage rounding is applied before independent scalar F32-activation math.
    oracle: dict = dict(vocab=vocab, context=context, cases=[dict(ids=ids, embedding=reference(ids)) for ids in cases], tolerance=5e-6)
    initial = [[struct.unpack("<f", struct.pack("<f", math.sin(row * hidden + col) * 0.3))[0]
                for col in range(hidden)] for row in range(7)]
    oracle["soft_cases"] = [dict(embeddings=initial, mask=[1] * len(initial),
                                 embedding=reference([1] * len(initial), initial))]
    oracle["text_cases"] = [dict(text=text, ids=ids, embedding=reference(ids)) for text, ids in
                            [("a", [2, 13, 1]), (" a", [2, 4, 1]), (" b", [2, 5, 1]), ("", [2, 1])]]
    if audio_soft is not None:
        hard = lambda token: [weight("token_embd")[token * hidden + i] * math.sqrt(hidden) for i in range(hidden)]
        merged = [hard(2), hard(vocab - 2)] + [list(row) for row in audio_soft] + [hard(vocab - 1), hard(1)]
        oracle["audio_embedding"] = reference([1] * len(merged), merged)
    return metadata, tensors, oracle


def run(directory, executable=None):
    directory.mkdir(parents=True, exist_ok=True)
    for storage in ["f32", "f16"]:
        for array_pattern in [False, True]:
            metadata, tensors, oracle = fixture(storage, array_pattern)
            stem = directory / (storage + ("-pattern-array" if array_pattern else ""))
            save(stem.with_suffix(".gguf"), metadata, tensors, storage)
            stem.with_suffix(".json").write_text(json.dumps(oracle), encoding="utf-8")
            if executable:
                subprocess.run([executable, str(stem.with_suffix(".gguf")), str(stem.with_suffix(".json"))], check=True)
    if executable:
        metadata, tensors, _ = fixture()
        cases = [("block_count", 0), ("embedding_length", 1.5), ("attention.head_count", [2, 2]),
                 ("attention.head_count_kv", [3, 1, 1]), ("attention.key_length_swa", 3),
                 ("attention.value_length", 4), ("rope.dimension_count", 9),
                 ("attention.sliding_window_pattern", [True, False]),
                 ("attention.layer_norm_rms_epsilon", -1.0), ("pooling_type", 2),
                 ("attention.causal", True), ("rope.scaling.type", "yarn"),
                 ("attention.shared_kv_layers", 1), ("rope.scaling.attn_factor", 2.0),
                 ("tokenizer.ggml.bos_token_id", -1), ("tokenizer.ggml.eos_token_id", True)]
        for i, (key, value) in enumerate(cases):
            invalid = copy.deepcopy(metadata)
            invalid[key if key.startswith("tokenizer.") else ARCH + "." + key] = value
            path = directory / f"invalid-{i}.gguf"
            save(path, invalid, tensors, "f32")
            subprocess.run([executable, str(path), "unused", "EmbeddingGemma2"], check=True)
        for name in ["blk.0.attn_q.weight", "blk.1.attn_k.weight", "per_layer_proj_norm.weight", "output.weight"]:
            invalid = dict(tensors)
            shape, values = invalid[name]
            invalid[name] = ([shape[0] + 1] + shape[1:], values + [0.0] * math.prod(shape[1:]))
            path = directory / "invalid-shape.gguf"
            save(path, metadata, invalid, "f32")
            subprocess.run([executable, str(path), "unused", name], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test")
    args = parser.parse_args()
    if args.output:
        run(args.output, args.test)
    else:
        with tempfile.TemporaryDirectory(prefix="embeddinggemma2-") as temp:
            run(Path(temp), args.test)
