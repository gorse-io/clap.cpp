#!/usr/bin/env python3
"""Published m-a-p/MERT-v2-FullSong F32 oracle at the pinned revision.

The source directory must be a local, checksum-verified snapshot. This is
deliberately separate from the random tiny F32/F16 architecture fixture.
"""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import torch
from safetensors.torch import load_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("build", type=Path)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--gguf", type=Path, help="verify an existing F32 GGUF instead of converting")
args = parser.parse_args()

revision = "7001f7657276238629e61c7746ce70735e0604c0"
expected_sha256 = "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d"
checkpoint_file = args.checkpoint / "model.safetensors"
digest = hashlib.sha256()
with checkpoint_file.open("rb") as source:
    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
        digest.update(block)
assert digest.hexdigest() == expected_sha256, digest.hexdigest()

args.output.mkdir(parents=True, exist_ok=True)
package = args.output / "official_mert2"
if package.exists():
    shutil.rmtree(package)
package.mkdir()
(package / "__init__.py").write_text("")
for name in ("configuration_mert2.py", "modeling_mert2.py"):
    shutil.copy2(args.checkpoint / name, package / name)
sys.path.insert(0, str(args.output))
configuration = importlib.import_module("official_mert2.configuration_mert2")
modeling = importlib.import_module("official_mert2.modeling_mert2")

raw = json.loads((args.checkpoint / "config.json").read_text())
assert raw["variant"] == "fs" and raw["context_seconds"] == 360.0
config = configuration.MERT2Config(**raw)
config._attn_implementation = "sdpa"
torch.set_num_threads(2)
with torch.device("meta"):
    model = modeling.MERT2Model(config)
state = load_file(checkpoint_file, device="cpu")
loaded = model.load_state_dict(state, assign=True, strict=True)
assert not loaded.missing_keys and not loaded.unexpected_keys, loaded
model.eval()

samples = config.minimum_input_samples
time = torch.arange(samples, dtype=torch.float32) / config.sampling_rate
waveform = 0.19 * torch.sin(2 * torch.pi * (220 * time + 31 * time * time))
waveform[-1] += 0.23
with torch.inference_mode():
    expected = model(waveform[None]).last_hidden_state[0].float()
assert expected.shape == (samples // config.inputs_to_logits_ratio, config.hidden_size)
assert torch.isfinite(expected).all()
fixture = {
    "published_checkpoint": True,
    "source_revision": revision,
    "checkpoint_sha256": expected_sha256,
    "sample_rate": config.sampling_rate,
    "hidden_size": config.hidden_size,
    "minimum_input_samples": config.minimum_input_samples,
    "inputs_to_logits_ratio": config.inputs_to_logits_ratio,
    "tolerance": 5e-4,
    "waveform_cases": [{
        "name": "published FullSong minimum waveform",
        "waveform": waveform.tolist(),
        "frames": expected.shape[0],
        "expected_frames": expected.flatten().tolist(),
        "expected_pooled": expected.mean(0).tolist(),
        "full_song": False,
    }],
}
fixture_path = args.output / "published-fixture.json"
fixture_path.write_text(json.dumps(fixture))
print("Published oracle", tuple(expected.shape), expected.flatten()[:4].tolist(), flush=True)

del model, state, expected
gc.collect()
converter = Path(__file__).resolve().parents[2] / "convert_hf_to_gguf.py"
gguf = args.gguf or args.output / "mert-v2-fullsong-f32.gguf"
if args.gguf is None:
    subprocess.run([sys.executable, str(converter), str(args.checkpoint), str(gguf),
                    "--mert2-revision", revision], check=True)
subprocess.run([str(args.build / "mert2-model-test"), str(gguf), str(fixture_path)], check=True)

# Exercise the public CLI once with the actual checkpoint.
raw_wave = struct.pack("<" + "f" * len(waveform), *waveform.tolist())
wav = args.output / "published-input.wav"
wav.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(raw_wave)) + b"WAVEfmt " +
                struct.pack("<IHHIIHH", 16, 3, 1, config.sampling_rate,
                            config.sampling_rate * 4, 4, 32) +
                b"data" + struct.pack("<I", len(raw_wave)) + raw_wave)
result = subprocess.run([str(args.build / "clap-model"), "--model", str(gguf), "--threads", "2",
                         "--audio", str(wav), "--frames"], text=True, capture_output=True)
assert result.returncode == 0, result.stderr
actual = json.loads(result.stdout)
expected_values = fixture["waveform_cases"][0]["expected_frames"]
error = max(abs(left - right) for left, right in zip(actual["values"], expected_values))
print(f"Published CLI frames={actual['frames']} max_abs_error={error:.9g}", flush=True)
assert actual["frames"] == 1 and actual["dimension"] == config.hidden_size and error <= fixture["tolerance"]
print("Published MERT-v2-FullSong checkpoint F32/frontend/native/CLI parity passed")
