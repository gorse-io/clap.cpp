#!/usr/bin/env python3
"""Independent tiny CLAP2 oracle: execute pinned official open_clip + timm.

Usage: python clap2_parity.py BUILD OUTPUT OFFICIAL_OPEN_CLIP HF_ASSETS
Use --generate-only to prepare fixtures before the native target is built.
No network/model downloads; only generated tiny weights (not checkpoint parity).
--profile xxs-head56 preserves the released XXS head dimension at reduced width.
--profile asymmetric-head64 exercises unequal tower widths and depths.
HF_ASSETS supplies the release config, open_clip_config.json and identity.json
(or legacy hf-manifest.json); --tokenizer-assets can reuse common tokenizer/header files.
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


def load_oracle_assets(assets):
    """Read the chosen release's official configuration and immutable identity.

    Kept stdlib-only so configuration/provenance checks need no Torch imports.
    The legacy full HF API manifest is accepted for pre-existing large assets.
    """
    config = json.loads((assets / "config.json").read_text())
    upstream = json.loads((assets / "open_clip_config.json").read_text())["model_cfg"]
    if (assets / "identity.json").exists():
        identity = json.loads((assets / "identity.json").read_text())
    else:
        manifest = json.loads((assets / "hf-manifest.json").read_text())
        identity = dict(repo=manifest["id"], revision=manifest["sha"])
    if not isinstance(identity.get("repo"), str) or not identity["repo"]:
        raise ValueError("oracle assets require a source repo")
    if not isinstance(identity.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", identity["revision"]):
        raise ValueError("oracle assets require a full immutable HF revision")
    comparisons = [("projection_dim", config["projection_dim"], upstream["embed_dim"])]
    for branch in ("audio", "text"):
        hf = config[branch + "_config"]
        official = (upstream["audio_cfg"]["naflexvit_cfg"] if branch == "audio" else upstream["text_cfg"])
        width, layers, heads = (("embed_dim", "depth", "num_heads") if branch == "audio" else
                               ("width", "layers", "heads"))
        for field, key in (("hidden_size", width), ("num_hidden_layers", layers), ("num_attention_heads", heads)):
            comparisons.append((f"{branch}.{key}", hf[field], official[key]))
        comparisons.append((f"{branch}.intermediate_size", hf["intermediate_size"],
                            int(official[width] * official["mlp_ratio"])))
    for label, hf_value, official_value in comparisons:
        if hf_value != official_value:
            raise ValueError(f"HF/open_clip configuration mismatch {label}: {hf_value} != {official_value}")
    return config, upstream, identity


def tiny_configs(config, upstream, profile):
    """Shrink official towers, retaining the selected head/layout edge cases."""
    # (audio width, text width, audio hidden, text hidden, audio heads, text heads,
    #  audio depth, text depth, projection width)
    profiles = {
        "tiny": (16, 16, 32, 32, 2, 2, 2, 2, 12),
        "asymmetric-head64": (128, 64, 352, 192, 2, 1, 2, 1, 48),
        "xxs-head56": (56, 56, 160, 160, 1, 1, 2, 2, 56),
    }
    aw, tw, ah, th, an, tn, al, tl, projection = profiles[profile]
    config, upstream = copy.deepcopy(config), copy.deepcopy(upstream)
    config["projection_dim"] = upstream["embed_dim"] = projection
    for branch, width, hidden, heads, layers in (("audio", aw, ah, an, al), ("text", tw, th, tn, tl)):
        config[branch + "_config"].update(hidden_size=width, intermediate_size=hidden,
                                        num_attention_heads=heads, num_hidden_layers=layers)
    config["audio_config"]["layerscale_value"] = 0.3
    config["text_config"]["max_position_embeddings"] = 32
    upstream["audio_cfg"]["naflexvit_cfg"].update(
        embed_dim=aw, depth=al, num_heads=an, mlp_ratio=ah / aw, init_values=0.3)
    upstream["text_cfg"].update(width=tw, heads=tn, layers=tl, mlp_ratio=th / tw, context_length=32)
    return config, upstream


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
    parser.add_argument("--profile", choices=("tiny", "asymmetric-head64", "xxs-head56"), default="tiny",
                        help="tiny tower shape contract; xxs preserves the released 56-wide RoPE heads")
    parser.add_argument("--tokenizer-assets", type=Path,
                        help="shared tokenizer files and HF tensor-name header (default: ASSETS)")
    args = parser.parse_args()
    tokenizer_assets = args.tokenizer_assets or args.assets
    source_config, source_upstream, identity = load_oracle_assets(args.assets)
    config, upstream_config = tiny_configs(source_config, source_upstream, args.profile)
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
    (args.output / "config.json").write_text(json.dumps(config))
    (args.output / "identity.json").write_text(json.dumps(identity))
    for name in ("processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        source = args.assets if name != "tokenizer.json" else tokenizer_assets
        shutil.copyfile(source / name, args.output / name)
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
    header = json.loads((tokenizer_assets / "model.safetensors.header.json").read_text())
    # Compare the complete reduced-depth HF key set (no renamed checkpoint tensors).
    expected_names = {name for name in header if name != "__metadata__" and
                      ((m := re.search(r"(audio|text)_model\.encoder\.layers\.(\d+)\.", name)) is None or
                       int(m[2]) < config[m[1] + "_config"]["num_hidden_layers"])}
    assert set(state) == expected_names, (set(state) - expected_names, expected_names - set(state))
    save_file(state, args.output / "model.safetensors")

    tokenizer = TikTokenTokenizer(encoding_name="r50k_base", context_length=32)
    hf_tokenizer = Tokenizer.from_file(str(tokenizer_assets / "tokenizer.json"))
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
        result = dict(source_revision=REVISION, hf_revision=identity["revision"], hf_repo=identity["repo"],
                      oracle_profile=args.profile, sample_rate=32000,
                      projection_dim=config["projection_dim"], context_length=32, tolerance=tolerance,
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
            case = dict(name=label, waveform_file=f"{label}.f32", samples=len(wave),
                        mel_frames=mel.shape[0], patches=patches["patches"].shape[0], expected=output.tolist())
            if args.profile == "asymmetric-head64" and label == "multiple":
                # This tonal fixture spans >130 dB: F32 MKL FFT residuals near
                # the -100 dB log floor differ from a double FFT. Injecting the
                # same official mel gives <3e-7 tower error; using official
                # double FFT reproduces native results. Bound this frontend-
                # sensitive case separately, not text, other waves or weights.
                case["tolerance"] = 2e-4
                case["precision_note"] = "F32 reference FFT versus native double FFT near the log-mel floor"
            result["waveform_cases"].append(case)
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
                        str(target), "--dtype", dtype, "--clap2-revision", identity["revision"]], check=True)
        if not args.generate_only:
            subprocess.run([str(args.build / "test-clap2-model"), str(target), str(fixture)], check=True)
        print(f"Official open_clip/timm CLAP2 {dtype} fixture passed; native={'not run' if args.generate_only else 'passed'}", flush=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "timm", "tiktoken", "safetensors", "tokenizers", "numpy")}
    (args.output / "oracle-versions.json").write_text(json.dumps(versions, indent=2))
    print(versions)


if __name__ == "__main__":
    main()
