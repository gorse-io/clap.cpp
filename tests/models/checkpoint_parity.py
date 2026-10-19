"""Verify actual local laion/larger_clap_music weights against Transformers.
No network access. Usage:
python tests/models/checkpoint_parity.py CHECKPOINT_DIR BUILD_DIR OUTPUT_DIR
The checkpoint must contain config.json, pytorch_model.bin, tokenizer.json and
preprocessor_config.json. Generates an oracle, then releases torch weights before
launching native inference. F32 tolerance is 5e-5 maximum absolute error.
"""
import gc
import json
from pathlib import Path
import subprocess
import sys
import struct

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import ClapConfig, ClapFeatureExtractor, ClapModel

source, build, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
config = ClapConfig.from_json_file(str(source / 'config.json'))
with torch.device('meta'):
    model = ClapModel(config)
state = torch.load(source / 'pytorch_model.bin', mmap=True, weights_only=True, map_location='cpu')
# Older checkpoints persist buffers that newer Transformers reconstruct. Never
# ignore missing parameters; ignoring only those legacy buffers is safe.
result = model.load_state_dict(state, assign=True, strict=False)
assert not result.missing_keys, result
assert all(k.endswith(('.position_ids', '.token_type_ids', '.relative_position_index')) for k in result.unexpected_keys), result
# Buffers removed from the modern state dict may still live on the meta device.
for module in model.modules():
    if hasattr(module, 'position_ids') and module.position_ids.device.type == 'meta':
        module.position_ids = torch.arange(config.text_config.max_position_embeddings).expand((1, -1))
model.eval()
tokenizer = Tokenizer.from_file(str(source / 'tokenizer.json'))
extractor = ClapFeatureExtractor.from_pretrained(str(source), local_files_only=True)
cases = []
with torch.no_grad():
    for prompt in ['A jazz quartet playing saxophone.', '你好，钢琴音乐 café 🎵', '', "I'm listening to heavy metal music."]:
        ids = tokenizer.encode(prompt).ids
        padded = ids + [1, 1]
        tokens = torch.tensor([padded])
        pooled = model.text_model(input_ids=tokens, attention_mask=(tokens != 1)).pooler_output
        expected = torch.nn.functional.normalize(model.text_projection(pooled), dim=-1)[0].tolist()
        cases.append(dict(text=prompt, ids=ids, padded_ids=padded, expected=expected))
    # Non-integral short length exercises repeatpad's zero-padded tail too.
    t = np.arange(67213, dtype=np.float64) / 48000
    waveform = (0.15 * np.sin(2 * np.pi * (220 * t + 170 * t * t)) +
                0.08 * np.cos(2 * np.pi * 880 * t)).astype(np.float32)
    features = extractor(waveform, sampling_rate=48000, return_tensors='pt')['input_features']
    pooled = model.audio_model(features).pooler_output
    expected = torch.nn.functional.normalize(model.audio_projection(pooled), dim=-1)[0].tolist()
fixture = dict(text_cases=cases, features=features.numpy().reshape(-1).tolist(), frames=features.shape[2],
               expected_audio=expected, waveform=waveform.tolist(),
               mel=features.numpy().reshape(-1).tolist(), end_to_end=True)
fixture_path = out / 'checkpoint-fixture.json'
fixture_path.write_text(json.dumps(fixture))
del model, state, pooled, features
# Trim freed allocations before spawning conversion/native inference on low RAM.
gc.collect()
root = Path(__file__).resolve().parents[2]
gguf = out / 'larger-clap-music-f32.gguf'
subprocess.run([sys.executable, str(root / 'src/models/convert_clap.py'), str(source), str(gguf)], check=True)
subprocess.run([str(build / 'clap-model-test'), str(gguf), str(fixture_path)], check=True)
# Exercise the public native executable, including its float32 WAV decoder.
raw = waveform.astype('<f4').tobytes()
wav = out / 'oracle.wav'
header = b'RIFF' + struct.pack('<I', 36 + len(raw)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 3, 1, 48000, 192000, 4, 32)
wav.write_bytes(header + b'data' + struct.pack('<I', len(raw)) + raw)
for args, expected in [(['--text', cases[0]['text']], cases[0]['expected']),
                       (['--audio', str(wav)], fixture['expected_audio'])]:
    p = subprocess.run([str(build / 'clap-model'), '--model', str(gguf), '--threads', '2', *args], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    actual = json.loads(p.stdout)
    assert len(actual) == 512
    error = max(abs(a - b) for a, b in zip(actual, expected))
    print(f'Public CLI {args[0]} max_abs_error={error:.9g}')
    assert error < 5e-5
print('Actual checkpoint F32 text/audio/tokenizer/frontend/end-to-end/CLI parity passed')
