#!/usr/bin/env python3
"""Deterministic tiny oracle using the pinned official MERT2 implementation.

This validates architecture/frontend parity only. Random tiny weights are not
evidence for the separately tested published checkpoint.
"""
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from safetensors.torch import save_file

build, out, reference = map(Path, sys.argv[1:])
root = Path(__file__).resolve().parents[2]
revision = "0c0c8c6f6e04dcc39d96f94e0be9978a74848906"

if out.exists():
    shutil.rmtree(out)
package = out / "official_mert2"
package.mkdir(parents=True)
(package / "__init__.py").write_text("")
for name in ("configuration_mert2.py", "modeling_mert2.py"):
    shutil.copy2(reference / name, package / name)
sys.path.insert(0, str(out))
configuration = importlib.import_module("official_mert2.configuration_mert2")
modeling = importlib.import_module("official_mert2.modeling_mert2")

torch.set_num_threads(2)
torch.manual_seed(20261019)
config = configuration.MERT2Config(
    hidden_size=8,
    intermediate_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_mel_bins=4,
    sampling_rate=64,
    n_fft=16,
    win_length=16,
    hop_length=4,
    subsampling_channels=[4, 8, 8],
    subsampling_depths=[1, 1, 1],
    conv_depthwise_kernel_size=3,
    variant="fs",
    context_seconds=360,
    attn_implementation="sdpa",
)
model = modeling.MERT2Model(config).eval()
config.to_json_file(out / "config.json")
(out / "preprocessor_config.json").write_text(json.dumps({
    "feature_extractor_type": "Wav2Vec2FeatureExtractor",
    "feature_size": 1,
    "sampling_rate": config.sampling_rate,
    "padding_value": 0.0,
    "padding_side": "right",
    "return_attention_mask": True,
    "do_normalize": False,
}))

def waveform(samples):
    time = torch.arange(samples, dtype=torch.float32) / config.sampling_rate
    value = 0.17 * torch.sin(2 * torch.pi * 7 * time) + 0.09 * torch.cos(2 * torch.pi * 13 * time)
    value[-1] += 0.31
    return value

@torch.no_grad()
def oracle(current, wave):
    output = current(wave[None]).last_hidden_state[0].float()
    return output, output.mean(0)

waves = [("short waveform", waveform(64), False),
         ("31.25-second full song", waveform(2000), True)]

def fixture_for(current, tolerance):
    cases = []
    for name, wave, full_song in waves:
        frames, pooled = oracle(current, wave)
        cases.append({"name": name, "waveform": wave.tolist(), "frames": frames.shape[0],
                      "expected_frames": frames.flatten().tolist(),
                      "expected_pooled": pooled.tolist(), "full_song": full_song})
    return {
        "source_revision": revision,
        "sample_rate": config.sampling_rate,
        "hidden_size": config.hidden_size,
        "minimum_input_samples": config.minimum_input_samples,
        "inputs_to_logits_ratio": config.inputs_to_logits_ratio,
        "tolerance": tolerance,
        "waveform_cases": cases,
    }

f32 = fixture_for(model, 2e-4)
(out / "fixture.json").write_text(json.dumps(f32))
save_file(model.state_dict(), out / "model.safetensors")

converter = root / "convert_hf_to_gguf.py"
common = ["--mert2-revision", revision]

# Match converter storage rounding. The official frontend deliberately remains
# float32 under model.half(), so its spectrogram/mel tensors remain exact.
rounded = modeling.MERT2Model(config).eval()
state = {}
for name, value in model.state_dict().items():
    state[name] = value.half().float() if value.ndim > 1 and not name.startswith("feature_extractor.") else value
rounded.load_state_dict(state, strict=True)
# F16 weights retain F32 activations; expose unintended F16 input packing.
f16 = fixture_for(rounded, 2e-5)
(out / "fixture-f16.json").write_text(json.dumps(f16))

# The official model code is shared: variant/context describe the checkpoint,
# not a cropping policy. Exercise the same >30-second waveform for both variants.
sys.path.insert(0, str(root))
from conversion.mert2 import mert2_inputs, write_safetensors_gguf
import numpy as np

for variant, context, model_name in (("fs", 360, "m-a-p/MERT-v2-FullSong"),
                                     ("30s", 30, "m-a-p/MERT-v2-30s")):
    current = out / "30s" if variant == "30s" else out
    current.mkdir(exist_ok=True)
    config.variant, config.context_seconds = variant, context
    config.to_json_file(current / "config.json")
    if current != out:
        for name in ("model.safetensors", "preprocessor_config.json", "fixture.json", "fixture-f16.json"):
            shutil.copy2(out / name, current / name)
    for dtype, suffix in (("f32", ""), ("f16", "-f16")):
        target = current / f"model{suffix}.gguf"
        subprocess.run([sys.executable, str(converter), str(current), str(target),
                        "--dtype", dtype, *common], check=True)
        checkpoint, metadata = mert2_inputs(current, revision)
        assert metadata["general.name"] == model_name, metadata
        assert json.loads(metadata["mert2.config"]) == json.loads((current / "config.json").read_text())
        assert metadata["mert2.source_revision"] == revision
        subprocess.run([str(build / "mert2-model-test"), str(target),
                        str(current / f"fixture{suffix}.json")], check=True)
    print(f"Synthetic official MERT2 {variant} F32/F16 and >30-second parity passed", flush=True)

# Retain unsupported-variant rejection, including the native loader. Bypass
# converter validation only to serialize an otherwise valid negative fixture.
raw = json.loads((out / "config.json").read_text())
raw["variant"] = "unsupported"
(out / "config.json").write_text(json.dumps(raw))
failed = subprocess.run([sys.executable, str(converter), str(out), str(out / "must-not-exist.gguf"), *common],
                        text=True, capture_output=True)
assert failed.returncode and "variant" in failed.stderr, failed.stderr
assert not (out / "must-not-exist.gguf").exists()
metadata["mert2.config"] = json.dumps(raw, separators=(",", ":"))
invalid = out / "invalid-variant.gguf"
write_safetensors_gguf(invalid, checkpoint, metadata, "f32", np, torch)
failed = subprocess.run([str(build / "mert2-model-test"), str(invalid), str(out / "fixture.json")],
                        text=True, capture_output=True)
assert failed.returncode and "variant" in failed.stderr, failed.stderr
print("Unsupported MERT2 variant rejected by converter and runtime")
