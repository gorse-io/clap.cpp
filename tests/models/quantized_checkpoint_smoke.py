"""Local published CLAP smoke; report loss, not a quality benchmark.

Usage: python tests/models/quantized_checkpoint_smoke.py BUILD CHECKPOINT F32 QUANTIZED OUTPUT.json
Requires the original local Transformers checkpoint. No downloads or uploads.
"""
import json
import pathlib
import subprocess
import sys
import wave
import numpy as np

build, checkpoint, f32, quantized, output = map(pathlib.Path, sys.argv[1:])
prompt = 'A jazz quartet playing saxophone.'
audio = output.with_suffix('.wav')
samples = (0.2 * np.sin(np.arange(48000) * 2 * np.pi * 440 / 48000) * 32767).astype('<i2')
with wave.open(str(audio), 'wb') as stream:
    stream.setnchannels(1)
    stream.setsampwidth(2)
    stream.setframerate(48000)
    stream.writeframes(samples.tobytes())
vectors, results = {}, {}
# Run native processes before loading Torch weights to keep peak RAM bounded.
for label, path in [('f32', f32), ('quantized', quantized)]:
    results[label] = {'path': str(path), 'bytes': path.stat().st_size}
    for mode, value in [('text', prompt), ('audio', str(audio))]:
        result = subprocess.run([str(build / 'clap'), 'embedding', '--model', str(path),
                                 '--threads', '2', '--' + mode, value],
                                capture_output=True, text=True, timeout=900)
        assert result.returncode == 0, result.stderr
        actual = np.asarray(json.loads(result.stdout))
        assert actual.shape == (512,) and np.isfinite(actual).all()
        norm = float(np.linalg.norm(actual))
        assert abs(norm - 1) < 1e-4
        vectors[label, mode] = actual
        results[label][mode] = {'dimension': len(actual), 'l2_norm': norm}
        print(label, mode, results[label][mode], flush=True)

import torch
from transformers import ClapConfig, ClapModel, ClapFeatureExtractor, RobertaTokenizerFast

torch.set_num_threads(2)
with torch.device('meta'):
    model = ClapModel(ClapConfig.from_pretrained(checkpoint, local_files_only=True))
state = torch.load(checkpoint / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
loaded = model.load_state_dict(state, assign=True, strict=False)
assert not loaded.missing_keys, loaded
assert all(key.endswith(('.position_ids', '.token_type_ids', '.relative_position_index'))
           for key in loaded.unexpected_keys), loaded
for module in model.modules():
    if hasattr(module, 'position_ids') and module.position_ids.device.type == 'meta':
        module.position_ids = torch.arange(model.config.text_config.max_position_embeddings).expand((1, -1))
model.eval()
tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
extractor = ClapFeatureExtractor.from_pretrained(checkpoint, local_files_only=True)
with torch.no_grad():
    pooled = model.text_model(**tokenizer(prompt, return_tensors='pt')).pooler_output
    text = torch.nn.functional.normalize(model.text_projection(pooled), dim=-1)[0].numpy()
    features = extractor(samples.astype(np.float32) / 32768, sampling_rate=48000,
                         return_tensors='pt')['input_features']
    pooled = model.audio_model(features).pooler_output
    audio_ref = torch.nn.functional.normalize(model.audio_projection(pooled), dim=-1)[0].numpy()
for label in results:
    for mode, expected in [('text', text), ('audio', audio_ref)]:
        actual = vectors[label, mode]
        cosine = float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        error = float(np.max(np.abs(actual - expected)))
        results[label][mode].update(transformers_cosine=cosine, max_abs_error=error)
        # Validate baseline implementation only. Report quantization loss without
        # inventing a quality threshold from one prompt and one synthetic tone.
        if label == 'f32':
            assert error < 5e-5, (mode, cosine, error)
        print(label, mode, 'Transformers', cosine, error, flush=True)
output.write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2), flush=True)
