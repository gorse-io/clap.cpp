#!/usr/bin/env python3
"""Tiny official-component MuQ F32/F16 waveform and CLI parity."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import torch
from safetensors.torch import save_file
from muq_reference import InferenceCore, MODEL_REVISION

build, out, reference = map(Path, sys.argv[1:])
root = Path(__file__).resolve().parents[2]
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(20261021)
raw = {
    'encoder_dim': 8, 'encoder_depth': 2, 'conv_dim': 4, 'n_mels': 128,
    'hop_length': 240, 'label_rate': 25, 'features': ['melspec_2048'], 'is_flash': False,
    'stat': {'melspec_2048_mean': 6.768444971712967, 'melspec_2048_std': 18.417922652295623},
    'w2v2_config': {'hidden_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,
                   'intermediate_size': 16, 'hidden_act': 'swish', 'position_embeddings_type': 'rotary',
                   'conv_depthwise_kernel_size': 3, 'rotary_embedding_base': 10000,
                   'num_conv_pos_embeddings': 8, 'num_conv_pos_embedding_groups': 2,
                   'layer_norm_eps': 1e-5, 'hidden_dropout': 0.0, 'attention_dropout': 0.0,
                   'activation_dropout': 0.0, 'conformer_conv_dropout': 0.0, 'layerdrop': 0.0},
}
model = InferenceCore(raw, reference).eval()
with torch.no_grad():
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.running_mean.uniform_(-0.3, 0.3)
            module.running_var.uniform_(0.4, 1.6)
            module.weight.uniform_(0.7, 1.3)
            module.bias.uniform_(-0.2, 0.2)
(out / 'config.json').write_text(json.dumps(raw))
save_file({'model.' + k: v for k, v in model.state_dict().items()}, out / 'model.safetensors')


def waveform(samples):
    time = torch.arange(samples, dtype=torch.float32) / 24000
    wave = 0.17 * torch.sin(2 * torch.pi * 220 * time) + 0.09 * torch.cos(2 * torch.pi * 733 * time)
    wave[-1] += 0.31
    return wave


waves = [('silence', torch.zeros(1025)), ('short', waveform(5041)),
         ('odd mel frames', waveform(5519)), ('31-second waveform', waveform(744000))]
for dtype, tolerance in [('f32', 3e-4), ('f16', 3e-3)]:
    current = copy.deepcopy(model)
    if dtype == 'f16':
        rounded = {name: value.half().float() if value.ndim > 1 else value
                   for name, value in current.state_dict().items()}
        current.load_state_dict(rounded, strict=True)
    cases = []
    with torch.inference_mode():
        for name, wave in waves:
            output = current(wave[None])[0].float()
            assert output.shape == ((len(wave) // 240 + 3) // 4, raw['encoder_dim'])
            cases.append({'name': name, 'waveform': wave.tolist(), 'frames': output.shape[0],
                          'expected_frames': output.flatten().tolist(), 'expected_pooled': output.mean(0).tolist()})
    fixture_path = out / ('fixture-' + dtype + '.json')
    fixture_path.write_text(json.dumps({'sample_rate': 24000, 'hidden_size': raw['encoder_dim'],
                                       'tolerance': tolerance, 'waveform_cases': cases}))
    gguf = out / ('model-' + dtype + '.gguf')
    subprocess.run([sys.executable, str(root / 'convert_hf_to_gguf.py'), str(out), str(gguf),
                    '--dtype', dtype, '--muq-revision', MODEL_REVISION], check=True)
    subprocess.run([str(build / 'muq-model-test'), str(gguf), str(fixture_path)], check=True)
    case = cases[2]
    data = struct.pack('<' + 'f' * len(case['waveform']), *case['waveform'])
    wav = out / 'input.wav'
    wav.write_bytes(b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' +
                    struct.pack('<IHHIIHH', 16, 3, 1, 24000, 96000, 4, 32) +
                    b'data' + struct.pack('<I', len(data)) + data)
    command = [str(build / 'clap-model'), '--model', str(gguf), '--threads', '2', '--audio', str(wav)]
    for frames in [False, True]:
        result = subprocess.run(command + (['--frames'] if frames else []), check=True,
                                capture_output=True, text=True, timeout=120)
        actual = json.loads(result.stdout)
        expected = case['expected_frames'] if frames else case['expected_pooled']
        if frames:
            assert actual['frames'] == case['frames'] and actual['dimension'] == raw['encoder_dim']
            actual = actual['values']
        assert len(actual) == len(expected)
        error = max(abs(a - b) for a, b in zip(actual, expected))
        assert error <= tolerance, (dtype, frames, error)
        print(f'MuQ {dtype} CLI frames={frames} max_abs_error={error:.9g}', flush=True)
    rejected = subprocess.run(command + ['--frames', '--crop-offset', '1'],
                              capture_output=True, text=True, timeout=120)
    assert rejected.returncode > 0 and 'crop' in rejected.stderr, rejected
print('MuQ official-component F32/F16, real-rate >30-second and CLI parity passed')
