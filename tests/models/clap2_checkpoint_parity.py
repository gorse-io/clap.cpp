#!/usr/bin/env python3
"""Low-memory published HF CLAP2 checkpoint oracle, one official tower/process.

python clap2_checkpoint_parity.py BRANCH CHECKPOINT_DIR OUTPUT OFFICIAL_OPEN_CLIP HF_ASSETS
BRANCH is text or audio. No downloads. Creates fixture-BRANCH-f32.json for
test-clap2-model; execute native separately after this process releases torch.
HF_ASSETS supplies config.json, open_clip_config.json and identity.json (or the
legacy hf-manifest.json). The official tower shape and revision come from those
assets, not the large model's source configuration.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from clap2_parity import REVISION, hf_name, hf_value, load_oracle_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=("text", "audio"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32",
                        help="storage-rounded oracle, always F32 activations")
    args = parser.parse_args()
    _, cfg, identity = load_oracle_assets(args.assets)
    assert (args.official / "PINNED_SHA").read_text().strip() == REVISION
    assert importlib.metadata.version("timm") == "1.0.30"
    sys.path.insert(0, str(args.official / "src"))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from open_clip.model import _build_text_tower
    from open_clip.audio.tower import AudioTower
    from open_clip.audio.naflex_audio import AudioNaFlexCfg, AudioNaFlexPatchify
    from open_clip.tokenizer import TikTokenTokenizer
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    # Instantiate only one tower; assign=True replaces initialized parameters
    # with mmap-backed safetensors views rather than cloning the full checkpoint.
    tower = (_build_text_tower(cfg["embed_dim"], cfg["text_cfg"]) if args.branch == "text" else
             AudioTower(cfg["audio_cfg"], cfg["embed_dim"]))
    with safe_open(args.checkpoint / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        # The gate/value half swap is its own inverse when importing HF weights.
        state = {}
        for name in tower.state_dict():
            value = hf_value(args.branch + "." + name,
                             checkpoint.get_tensor(hf_name(args.branch + "." + name)))
            state[name] = value.half().float() if args.dtype == "f16" and value.ndim == 2 else value
        tower.load_state_dict(state, strict=True, assign=True)
    tower.eval()
    fixture = dict(source_revision=REVISION, hf_revision=identity["revision"], hf_repo=identity["repo"], sample_rate=32000,
                   projection_dim=cfg["embed_dim"], context_length=cfg["text_cfg"]["context_length"],
                   tolerance=8e-5, text_cases=[], token_cases=[], waveform_cases=[])
    with torch.no_grad():
        if args.branch == "text":
            tok = TikTokenTokenizer("r50k_base", context_length=cfg["text_cfg"]["context_length"])
            for text in ("", "a dog barking", "We're I'M can't CAN'T", "café 中文 🙂", "a  b\t\nnext  ",
                         "<|endoftext|> <|bos|> <|pad|>"):
                ids = tok(text)[0]
                expected = F.normalize(tower(ids[None]), dim=-1)[0]
                public_ids = tok(text, pad=False)[0]
                torch.testing.assert_close(expected, F.normalize(tower(public_ids[None]), dim=-1)[0],
                                           atol=2e-6, rtol=2e-6)
                fixture["text_cases"].append(dict(text=text, tokens=public_ids.tolist(), expected=expected.tolist()))
        else:
            patchify = AudioNaFlexPatchify(AudioNaFlexCfg.from_clip_audio_cfg(tower.cfg), max_audio_tokens=None)
            for name, length in (("checkpoint-subpatch", 600), ("checkpoint-nonmultiple", 5137),
                                 ("checkpoint-one-second", 32000)):
                t = np.arange(length, dtype=np.float64) / 32000
                wave = (0.17 * np.sin(2 * np.pi * 437 * t) + 0.09 * np.cos(2 * np.pi * 1337 * t)).astype(np.float32)
                wave[-1] += 0.31
                wave.astype("<f4").tofile(args.output / f"{name}.f32")
                patches = patchify((torch.from_numpy(wave), 32000))
                expected = F.normalize(tower({key: value[None] for key, value in patches.items()}), dim=-1)[0]
                fixture["waveform_cases"].append(dict(name=name, samples=length, waveform_file=f"{name}.f32",
                                                       expected=expected.tolist()))
    path = args.output / f"fixture-{args.branch}-{args.dtype}.json"
    path.write_text(json.dumps(fixture))
    print(f"Published checkpoint official {args.branch} {args.dtype} oracle generated: {path}")


if __name__ == "__main__":
    main()
