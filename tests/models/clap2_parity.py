#!/usr/bin/env python3
"""Independent tiny CLAP2 oracle: execute pinned official open_clip + timm.

Usage: python clap2_parity.py BUILD OUTPUT OFFICIAL_OPEN_CLIP HF_ASSETS
Use --generate-only to prepare fixtures before the native target is built.
No network/model downloads; only generated tiny weights (not checkpoint parity).
"""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

REVISION = "f9a0948d573009f8fa3e530e55814cb11fb8e8bd"
HF_REVISION = "e836feaa79cd1f7de939fe77065819f48fc44e4c"


def hf_name(name):
    """Name-only bridge, independent of production conversion/inference code."""
    exact = {
        "logit_scale": "logit_scale",
        "audio.encoder.vit.embeds.reg_token": "audio_model.embeddings.register_tokens",
        "audio.encoder.vit.embeds.proj.weight": "audio_model.embeddings.patch_embedding.weight",
        "audio.encoder.vit.norm_pre.weight": "audio_model.embeddings.layernorm.weight",
        "audio.encoder.vit.norm.weight": "audio_model.norm.weight",
        "text.token_embedding.weight": "text_model.embed_tokens.weight",
        "text.ln_final.weight": "text_model.norm.weight",
        "text.text_projection.weight": "text_projection.weight",
    }
    if name in exact:
        return exact[name]
    if name.startswith("audio.proj."):
        return name.replace("audio.proj.0.", "audio_projection.linear1.").replace(
            "audio.proj.2.", "audio_projection.linear2.")
    if name.startswith("audio.encoder.vit.attn_pool."):
        tail = name.removeprefix("audio.encoder.vit.attn_pool.")
        for source, target in (("latent", "probe"), ("q.", "q_proj."), ("kv.", "kv_proj."),
                               ("proj.", "o_proj."), ("norm.", "layernorm."),
                               ("mlp.fc1.", "fc1."), ("mlp.fc2.", "fc2.")):
            if tail == source or tail.startswith(source):
                return "audio_model.head." + target + tail[len(source):]
    if name.startswith("text.pool."):
        tail = name.removeprefix("text.pool.")
        for source, target in (("query", "probe"), ("q.", "q_proj."), ("kv.", "kv_proj."),
                               ("q_norm.", "q_norm."), ("k_norm.", "k_norm.")):
            if tail == source or tail.startswith(source):
                return "text_model.head." + target + tail[len(source):]
    match = re.fullmatch(r"(audio\.encoder\.vit|text)\.blocks\.(\d+)\.(.*)", name)
    if match:
        tower, layer, tail = match.groups()
        prefix = "audio_model" if tower.startswith("audio") else "text_model"
        replacements = (("gamma_1", "layer_scale1.lambda1"), ("gamma_2", "layer_scale2.lambda1"),
                        ("norm1.", "input_layernorm."), ("norm2.", "post_attention_layernorm."),
                        ("attn.qkv.", "self_attn.qkv_proj."), ("attn.proj.", "self_attn.o_proj."),
                        ("attn.q_norm.", "self_attn.q_norm."), ("attn.k_norm.", "self_attn.k_norm."),
                        ("mlp.fc1.", "mlp.gate_up_proj."), ("mlp.fc2.", "mlp.down_proj."),
                        ("mlp.w12.", "mlp.gate_up_proj."), ("mlp.w3.", "mlp.down_proj."))
        for source, target in replacements:
            if tail == source or tail.startswith(source):
                return f"{prefix}.encoder.layers.{layer}." + target + tail[len(source):]
    raise AssertionError(f"unmapped upstream parameter: {name}")


def hf_value(name, value):
    """Official text SwiGLU stores [value, gate]; HF stores [gate, value].

    Confirmed against range-read bytes of the pinned published open_clip and HF
    checkpoints, not inferred just from the tensor's name. Audio timm SwiGLU is
    already gate-first and QKV/pool tensors have identical layouts.
    """
    if re.fullmatch(r"text\.blocks\.\d+\.mlp\.w12\.(weight|bias)", name):
        first, second = value.chunk(2, dim=0)
        import torch
        return torch.cat((second, first), dim=0)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assert (args.official / "PINNED_SHA").read_text().strip() == REVISION
    sys.path.insert(0, str(args.official / "src"))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import open_clip
    from open_clip.tokenizer import TikTokenTokenizer
    from open_clip.audio.dsp import log_mel_spectrogram
    from open_clip.audio.naflex_audio import AudioNaFlexCfg, AudioNaFlexPatchify, mel_to_patches
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    assert Path(open_clip.__file__).resolve().is_relative_to(args.official.resolve())
    assert importlib.metadata.version("timm") == "1.0.30", "oracle requires pinned timm 1.0.30"
    torch.set_num_threads(2)
    torch.manual_seed(20261107)
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.assets / "config.json").read_text())
    config["projection_dim"] = 12
    for tower in ("audio_config", "text_config"):
        config[tower].update(hidden_size=16, intermediate_size=32, num_attention_heads=2, num_hidden_layers=2)
    config["audio_config"]["layerscale_value"] = 0.3
    config["text_config"]["max_position_embeddings"] = 32
    (args.output / "config.json").write_text(json.dumps(config))
    for name in ("processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(args.assets / name, args.output / name)
    upstream_config = json.loads((args.official / "src/open_clip/model_configs/naflexclap_large.json").read_text())
    upstream_config["embed_dim"] = config["projection_dim"]
    upstream_config["audio_cfg"]["naflexvit_cfg"].update(
        embed_dim=16, depth=2, num_heads=2, mlp_ratio=2.0, init_values=0.3)
    upstream_config["text_cfg"].update(width=16, heads=2, layers=2, mlp_ratio=2.0, context_length=32)
    model = open_clip.CLAP(**upstream_config).eval()
    # Nontrivial biases/scales prevent zero/default initialization hiding bugs.
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "norm" in name or "ln_final" in name:
                value.copy_(0.8 + 0.2 * torch.rand_like(value))
            elif "gamma_" in name:
                value.copy_(0.2 + 0.2 * torch.rand_like(value))
            else:
                value.normal_(std=0.12 if value.ndim > 1 else 0.05)
    state = {hf_name(name): hf_value(name, value).detach().clone().contiguous()
             for name, value in model.state_dict().items()}
    header = json.loads((args.assets / "model.safetensors.header.json").read_text())
    # Compare the complete reduced-depth HF key set (no renamed checkpoint tensors).
    expected_names = {name for name in header if name != "__metadata__" and
                      ((m := re.search(r"\.layers\.(\d+)\.", name)) is None or int(m[1]) < 2)}
    assert set(state) == expected_names, (set(state) - expected_names, expected_names - set(state))
    save_file(state, args.output / "model.safetensors")

    tokenizer = TikTokenTokenizer(encoding_name="r50k_base", context_length=32)
    hf_tokenizer = Tokenizer.from_file(str(args.assets / "tokenizer.json"))
    hf_tokenizer.encode_special_tokens = True  # split_special_tokens / encode_ordinary contract
    texts = ["", "a dog barking", "We're I'M can't CAN'T I'd i'd", "a  b   c\t\nend  ",
             "café CAFÉ e\u0301 中文 🙂", "1234567890 ² Ⅷ", "line\r\n\t next\n\n", "word " * 50,
             " punctuation...!? -- _snake_case", "\u00a0space\u2003test",
             "<|endoftext|> <|bos|> <|eos|> <|pad|>"]
    text_inputs = []
    for text in texts:
        tokens = tokenizer(text)[0]
        hf_ids = hf_tokenizer.encode(text).ids
        # HF JSON describes the same body BPE. Compare before explicit truncation.
        ordinary = tokenizer.enc.encode_ordinary(text)
        assert hf_ids == [50259] + ordinary + [50257], (text, hf_ids, ordinary)
        text_inputs.append((text, tokens))
    # Explicit IDs expose padding placement, bidirectionality and masked MAP pooling.
    token_cases = [("bos-eos", [50259, 50257]), ("interior pad", [50259, 123, 50258, 456, 50257]),
                   ("pad suffix", [50259, 123, 456, 50257] + [50258] * 5),
                   ("no eos", [50259, 123, 456]), ("reordered", [50259, 456, 123, 50257])]
    wave_inputs = []
    patchify = AudioNaFlexPatchify(AudioNaFlexCfg.from_clip_audio_cfg(model.audio.cfg), max_audio_tokens=None)
    for label, length in (("subpatch", 600), ("nonmultiple", 5137), ("multiple", 4800),
                          ("long-no-autocrop", 32_000 * 31 + 137), ("silence", 600)):
        t = np.arange(length, dtype=np.float64) / 32000
        wave = (0.17 * np.sin(2 * np.pi * 437 * t) + 0.09 * np.cos(2 * np.pi * 1337 * t)).astype(np.float32)
        if label == "silence":
            wave.fill(0)
        elif label == "long-no-autocrop":
            # A nonstationary tail makes a silent 30-second crop numerically visible.
            tail = t >= 30
            wave[tail] = (0.8 * np.sin(2 * np.pi * 7300 * t[tail])).astype(np.float32)
            wave[-1] += 0.31
        else:
            wave[-1] += 0.31
        path = args.output / f"{label}.f32"
        wave.astype("<f4").tofile(path)
        samples = torch.from_numpy(wave)
        patches = patchify((samples, 32000))  # actual upstream sub-window padding + frontend + patchify
        mel = log_mel_spectrogram(F.pad(samples, (0, max(0, 1024 - length))),
                                 32000, 1024, 320, 128, 50, 16000).T
        torch.testing.assert_close(patches["patches"], mel_to_patches(mel, patch_freq=16, patch_time=16)["patches"])
        assert patches["patches"].shape[0] == 8 * ((mel.shape[0] + 15) // 16)
        wave_inputs.append((label, wave, patches, mel))

    @torch.no_grad()
    def fixtures(current, tolerance):
        result = dict(source_revision=REVISION, hf_revision=HF_REVISION, sample_rate=32000,
                      projection_dim=12, context_length=32, tolerance=tolerance,
                      text_cases=[], token_cases=[], waveform_cases=[])
        for text, ids in text_inputs:
            output = current.encode_text(ids[None], normalize=True)[0]
            # Public tokenize returns variable-length IDs; padding is an encoder
            # batch policy, not part of the C API token count.
            public_ids = tokenizer(text, pad=False)[0]
            torch.testing.assert_close(output, current.encode_text(public_ids[None], normalize=True)[0],
                                       atol=2e-6, rtol=2e-6)
            result["text_cases"].append(dict(text=text, tokens=public_ids.tolist(), expected=output.tolist()))
        for label, ids in token_cases:
            output = current.encode_text(torch.tensor([ids]), normalize=True)[0]
            result["token_cases"].append(dict(name=label, tokens=ids, expected=output.tolist()))
        compact = torch.tensor([[50259, 123, 456, 50257]])
        padded = F.pad(compact, (0, 20), value=50258)
        torch.testing.assert_close(current.encode_text(compact), current.encode_text(padded), atol=2e-6, rtol=2e-6)
        for label, wave, patches, mel in wave_inputs:
            batch = {key: value[None] for key, value in patches.items()}
            output = current.encode_audio(batch, normalize=True)[0]
            if label == "long-no-autocrop":
                cropped_mel = log_mel_spectrogram(torch.from_numpy(wave[:32000 * 30]),
                                                32000, 1024, 320, 128, 50, 16000).T
                cropped = mel_to_patches(cropped_mel, patch_freq=16, patch_time=16)
                cropped_output = current.encode_audio({key: value[None] for key, value in cropped.items()},
                                                       normalize=True)[0]
                crop_error = (output - cropped_output).abs().max().item()
                assert crop_error > 5 * tolerance, f"long fixture cannot detect 30s crop: {crop_error}"
                print(f"30-second autocrop negative control max_abs_error={crop_error:.6g}", flush=True)
            # Invalid patch slots must not enter attention or attentive pooling.
            padded_batch = {key: F.pad(value, (0, 0, 0, 3)) if value.ndim == 3 else F.pad(value, (0, 3))
                            for key, value in batch.items()}
            padded_batch["patches"][:, -3:] = 999
            torch.testing.assert_close(current.encode_audio(batch), current.encode_audio(padded_batch), atol=2e-6, rtol=2e-6)
            result["waveform_cases"].append(dict(name=label, waveform_file=f"{label}.f32", samples=len(wave),
                mel_frames=mel.shape[0], patches=patches["patches"].shape[0], expected=output.tolist()))
        return result

    for dtype in ("f32", "f16"):
        current = copy.deepcopy(model)
        if dtype == "f16":
            # Converter keeps learned register/probe vectors and 1D/scalar data F32.
            rounded = {name: value.half().float() if value.ndim == 2 else value
                       for name, value in current.state_dict().items()}
            current.load_state_dict(rounded, strict=True)
        fixture = args.output / f"fixture-{dtype}.json"
        fixture.write_text(json.dumps(fixtures(current, 5e-5)))
        target = args.output / f"model-{dtype}.gguf"
        subprocess.run([sys.executable, str(root / "convert_hf_to_gguf.py"), str(args.output),
                        str(target), "--dtype", dtype], check=True)
        if not args.generate_only:
            subprocess.run([str(args.build / "clap2-model-test"), str(target), str(fixture)], check=True)
        print(f"Official open_clip/timm CLAP2 {dtype} fixture passed; native={'not run' if args.generate_only else 'passed'}", flush=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "timm", "tiktoken", "safetensors", "tokenizers", "numpy")}
    (args.output / "oracle-versions.json").write_text(json.dumps(versions, indent=2))
    print(versions)


if __name__ == "__main__":
    main()
