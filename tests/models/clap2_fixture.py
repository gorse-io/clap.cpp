#!/usr/bin/env python3
"""Stdlib-only valid CLAP2 GGUF for default CTest loader/API smoke tests.

This analytical degenerate fixture is NOT the optional official numerical oracle.
No source/model downloads or optional Python ML packages are required.
"""
import json
import math
from pathlib import Path
import struct


def byte_encoder():
    printable = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    missing = [byte for byte in range(256) if byte not in printable]
    return {**{byte: chr(byte) for byte in range(256) if byte not in missing},
            **{byte: chr(256 + index) for index, byte in enumerate(missing)}}


def model_data():
    common = dict(hidden_size=8, intermediate_size=16, num_attention_heads=1, num_hidden_layers=1,
                  hidden_act="silu", rms_norm_eps=1e-6, attention_dropout=0.0)
    audio = dict(common, model_type="clap2_audio_model", attention_bias=True, mlp_bias=True,
                 use_qk_norm=False, layerscale_value=0.3, num_mel_bins=128, num_register_tokens=1,
                 patch_size=16, pooler_hidden_act="gelu", rope_parameters=dict(rope_type="axial", rope_theta=10000.0))
    text = dict(common, model_type="clap2_text_model", attention_bias=False, mlp_bias=False,
                use_qk_norm=True, layerscale_value=None, vocab_size=260, max_position_embeddings=32,
                bos_token_id=259, eos_token_id=257, pad_token_id=258,
                rope_parameters=dict(rope_type="default", rope_theta=10000.0))
    config = dict(architectures=["Clap2Model"], model_type="clap2", projection_dim=4,
                  projection_hidden_act="gelu", audio_config=audio, text_config=text)
    processor = dict(processor_class="Clap2Processor", feature_extractor=dict(
        feature_extractor_type="Clap2FeatureExtractor", sampling_rate=32000, n_fft=1024,
        hop_length=320, feature_size=128, patch_size=16, f_min=50.0, f_max=16000.0,
        max_num_patches=None, padding_side="right", padding_value=0.0, return_attention_mask=True))
    vocab = {char: byte for byte, char in byte_encoder().items()}
    added = []
    for number, name in enumerate(("<|endoftext|>", "<|eos|>", "<|pad|>", "<|bos|>"), 256):
        vocab[name] = number
        added.append(dict(id=number, content=name, single_word=False, lstrip=False, rstrip=False,
                          normalized=False, special=True))
    tokenizer = dict(version="1.0", normalizer=None, added_tokens=added,
        pre_tokenizer=dict(type="ByteLevel", add_prefix_space=False, trim_offsets=True, use_regex=True),
        decoder=dict(type="ByteLevel", add_prefix_space=True, trim_offsets=True, use_regex=True),
        model=dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix="", end_of_word_suffix="",
                   fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab=vocab, merges=[]),
        post_processor=dict(type="TemplateProcessing", single=[
            {"SpecialToken": {"id": "<|bos|>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<|eos|>", "type_id": 0}}], special_tokens={
                token: dict(id=token, ids=[number], tokens=[token])
                for token, number in (("<|bos|>", 259), ("<|eos|>", 257))}))
    tensors = {}

    def add(name, shape, values=None):
        count = math.prod(shape)
        tensors[name] = (shape, [0.0] * count if values is None else list(values))

    def norm(name):
        add(name + ".weight", (8,), [1.0] * 8)

    def linear(name, output, width, bias=False):
        add(name + ".weight", (output, width))
        if bias:
            add(name + ".bias", (output,))

    add("logit_scale", ())
    add("audio_model.embeddings.register_tokens", (1, 1, 8))
    linear("audio_model.embeddings.patch_embedding", 8, 256)
    norm("audio_model.embeddings.layernorm")
    add("text_model.embed_tokens.weight", (260, 8), [1.0, 0, 0, 0, 0, 0, 0, 0] * 260)
    for branch in ("audio", "text"):
        prefix = f"{branch}_model.encoder.layers.0."
        norm(prefix + "input_layernorm")
        norm(prefix + "post_attention_layernorm")
        linear(prefix + "self_attn.qkv_proj", 24, 8, branch == "audio")
        linear(prefix + "self_attn.o_proj", 8, 8, branch == "audio")
        linear(prefix + "mlp.gate_up_proj", 32, 8, branch == "audio")
        linear(prefix + "mlp.down_proj", 8, 16, branch == "audio")
        if branch == "audio":
            for index in (1, 2):
                add(prefix + f"layer_scale{index}.lambda1", (8,), [0.3] * 8)
        else:
            norm(prefix + "self_attn.q_norm")
            norm(prefix + "self_attn.k_norm")
        norm(f"{branch}_model.norm")
        head = f"{branch}_model.head."
        add(head + "probe", (1, 1, 8))
        linear(head + "q_proj", 8, 8, branch == "audio")
        linear(head + "kv_proj", 16, 8, branch == "audio")
        if branch == "audio":
            linear(head + "o_proj", 8, 8, True)
            norm(head + "layernorm")
            linear(head + "fc1", 16, 8, True)
            linear(head + "fc2", 8, 16, True)
        else:
            norm(head + "q_norm")
            norm(head + "k_norm")
            values = [0.0] * 128
            for index in range(8):
                values[(8 + index) * 8 + index] = 1.0
            add(head + "kv_proj.weight", (16, 8), values)
    linear("text_projection", 4, 8)
    tensors["text_projection.weight"][1][0] = 1.0
    linear("audio_projection.linear1", 4, 8, True)
    linear("audio_projection.linear2", 4, 4, True)
    tensors["audio_projection.linear2.bias"][1][1] = 1.0
    return config, processor, tokenizer, tensors


def write_gguf(path, metadata, tensors, half=False):
    def string(value):
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        header += string(key) + struct.pack("<I", 8) + string(value)
    body = bytearray()
    for name, (shape, values) in sorted(tensors.items()):
        dims = shape or (1,)
        is_half = half and len(shape) == 2
        header += string(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *reversed(dims))
        header += struct.pack("<IQ", int(is_half), len(body))
        body += struct.pack("<" + ("e" if is_half else "f") * len(values), *values)
        body += b"\0" * (-len(body) % 32)
    path.write_bytes(header + b"\0" * (-len(header) % 32) + body)


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    config, processor, tokenizer, tensors = model_data()
    metadata = {"general.architecture": "clap2", "general.name": "tiny-analytical-clap2",
                "clap2.config": json.dumps(config), "clap2.processor_config": json.dumps(processor),
                "clap2.tokenizer": json.dumps(tokenizer)}
    for half in (False, True):
        write_gguf(output / ("model-f16.gguf" if half else "model-f32.gguf"), metadata, tensors, half)
    fixture = dict(sample_rate=32000, projection_dim=4, context_length=32, tolerance=2e-6,
                   text_cases=[], token_cases=[], waveform_cases=[])
    for text in ("", "hello", "café 中文 🙂", "a  b\t\n", "<|bos|>"):
        ids = ([259] + list(text.encode("utf-8")) + [257])
        fixture["text_cases"].append(dict(text=text, tokens=ids, expected=[1, 0, 0, 0]))
    for name, ids in (("right pad", [259, 12, 257, 258]), ("interior pad", [259, 258, 12, 257])):
        fixture["token_cases"].append(dict(name=name, tokens=ids, expected=[1, 0, 0, 0]))
    wave = [0.1 * math.sin(index * 0.17) for index in range(600)]
    (output / "smoke.f32").write_bytes(struct.pack("<" + "f" * len(wave), *wave))
    fixture["waveform_cases"].append(dict(name="smoke", waveform_file="smoke.f32", samples=len(wave), expected=[0, 1, 0, 0]))
    (output / "fixture.json").write_text(json.dumps(fixture))
    return metadata, tensors


if __name__ == "__main__":
    import sys
    generate(Path(sys.argv[1]))
