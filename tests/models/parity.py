"""Deterministic Transformers oracle; Python is used for conversion/tests only.
Usage: python tests/models/parity.py BUILD_DIR FIXTURE_DIR
Requires torch, transformers, numpy. The native runner consumes JSON, not Python.
"""
import json
import pathlib
import subprocess
import struct
import sys
import numpy as np
import torch
from transformers import ClapConfig, ClapModel, ClapFeatureExtractor

ROOT = pathlib.Path(__file__).resolve().parents[2]
build, out = map(pathlib.Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(123)
c = ClapConfig(
    text_config=dict(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                     num_hidden_layers=2, vocab_size=50265, projection_dim=12),
    audio_config=dict(hidden_size=64, patch_embeds_hidden_size=8,
                      num_attention_heads=[1, 2, 4, 8], depths=[2, 2, 2, 2],
                      num_mel_bins=32, spec_size=128, window_size=2, projection_dim=12),
    projection_dim=12)
m = ClapModel(c).eval()
(out / 'config.json').write_text(c.to_json_string())
torch.save(m.state_dict(), out / 'pytorch_model.bin')
# Byte-level tokens and merges exercising Unicode and contractions.
bytechars = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
extra = 256
mapping = {}
for b in range(256):
    if b in bytechars: mapping[b] = chr(b)
    else: mapping[b] = chr(extra); extra += 1
vocab = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '<mask>': 4}
for ch in mapping.values(): vocab[ch] = len(vocab)
merges = ['h e', 'he l', 'hel l', 'hell o']
for merge in merges: vocab[merge.replace(' ', '')] = len(vocab)
tok = dict(model=dict(type='BPE', vocab=vocab, merges=merges),
           added_tokens=[dict(id=vocab[t], content=t, special=True, lstrip=t=='<mask>', rstrip=False) for t in list(vocab)[:5]])
(out / 'tokenizer.json').write_text(json.dumps(tok))
from tokenizers import Tokenizer, models, pre_tokenizers, processors, AddedToken
tokenizer = Tokenizer(models.BPE(vocab, [tuple(x.split()) for x in merges], unk_token='<unk>'))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.post_processor = processors.RobertaProcessing(('</s>', 2), ('<s>', 0), add_prefix_space=False)
tokenizer.add_special_tokens([AddedToken(x['content'], special=True, lstrip=x['lstrip']) for x in tok['added_tokens']])
prompts = ['hello world!', "I'm testing 你好 café\n\t", 'x  <mask> y', '<s>hello</s>']
cases = []
with torch.no_grad():
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        padded = ids + [1, 1]
        ids_t = torch.tensor([padded])
        pooled = m.text_model(input_ids=ids_t, attention_mask=(ids_t != 1)).pooler_output
        expected = torch.nn.functional.normalize(m.text_projection(pooled), dim=-1)[0].tolist()
        cases.append(dict(text=prompt, ids=ids, padded_ids=padded, expected=expected))
    features = (np.random.default_rng(123).standard_normal((1, 1, 501, 32)) * 3 - 30).astype(np.float32)
    pooled = m.audio_model(torch.from_numpy(features)).pooler_output
    expected_audio = torch.nn.functional.normalize(m.audio_projection(pooled), dim=-1)[0].tolist()
wave = (0.2 * np.sin(np.arange(24000) * (2 * np.pi * 440 / 48000))).astype(np.float32)
extractor = ClapFeatureExtractor(truncation='rand_trunc', padding='repeatpad', frequency_min=50)
mel = extractor(wave, sampling_rate=48000, return_tensors='np')['input_features'][0, 0]
fixture = dict(text_cases=cases, features=features.reshape(-1).tolist(), frames=501,
               expected_audio=expected_audio, waveform=wave.tolist(), mel=mel.reshape(-1).tolist())
(out / 'fixture.json').write_text(json.dumps(fixture))
subprocess.run([sys.executable, str(ROOT / 'convert_hf_to_gguf.py'), str(out), str(out / 'model.gguf')], check=True)
subprocess.run([str(build / 'clap-model-test'), str(out / 'model.gguf'), str(out / 'fixture.json')], check=True)
# F16 storage parity against the same parameters rounded exactly as conversion.
with torch.no_grad():
    for parameter in m.parameters():
        if parameter.ndim > 1:
            parameter.copy_(parameter.half().float())
    rounded = json.loads(json.dumps(fixture))
    # ggml's F16 dot path also rounds activations to F16, unlike this F32 oracle.
    rounded['embedding_tolerance'] = 1e-3
    for case in rounded['text_cases']:
        tokens = torch.tensor([case['padded_ids']])
        pooled = m.text_model(input_ids=tokens, attention_mask=(tokens != 1)).pooler_output
        case['expected'] = torch.nn.functional.normalize(m.text_projection(pooled), dim=-1)[0].tolist()
    pooled = m.audio_model(torch.from_numpy(features)).pooler_output
    rounded['expected_audio'] = torch.nn.functional.normalize(m.audio_projection(pooled), dim=-1)[0].tolist()
(out / 'fixture-f16.json').write_text(json.dumps(rounded))
subprocess.run([sys.executable, str(ROOT / 'convert_hf_to_gguf.py'), str(out), str(out / 'model-f16.gguf'), '--dtype', 'f16'], check=True)
subprocess.run([str(build / 'clap-model-test'), str(out / 'model-f16.gguf'), str(out / 'fixture-f16.json')], check=True)
# Q8 native inference against exact stored weights, not an unquantized oracle.
from q8_reference import load_storage
sys.path.insert(0, str(ROOT))
from convert_hf_to_gguf import clap_tensor_name
for source, suffix in [('model.gguf', 'q8'), ('model-f16.gguf', 'f16-q8')]:
    target = out / f'model-{suffix}.gguf'
    if target.exists(): target.unlink()
    subprocess.run([str(build / 'clap-quantize'), str(out / source), str(target)], check=True)
    count = load_storage(m, target, clap_tensor_name)
    quantized_fixture = json.loads(json.dumps(fixture))
    # Block packing is discontinuous: small interpolation/normalization rounding
    # differences can move audio activations across Q8 bins. Allow 1% absolute
    # normalized-embedding error; the C++ test independently checks exact bytes
    # and native matmul against canonical dequantized weights.
    quantized_fixture['embedding_tolerance'] = 1e-2
    with torch.no_grad():
        for case in quantized_fixture['text_cases']:
            tokens = torch.tensor([case['padded_ids']])
            pooled = m.text_model(input_ids=tokens, attention_mask=(tokens != 1)).pooler_output
            case['expected'] = torch.nn.functional.normalize(m.text_projection(pooled), dim=-1)[0].tolist()
        pooled = m.audio_model(torch.from_numpy(features)).pooler_output
        quantized_fixture['expected_audio'] = torch.nn.functional.normalize(m.audio_projection(pooled), dim=-1)[0].tolist()
    fixture_path = out / f'fixture-{suffix}.json'
    fixture_path.write_text(json.dumps(quantized_fixture))
    subprocess.run([str(build / 'clap-model-test'), str(target), str(fixture_path)], check=True)
    print(f'CLAP {suffix}: {count} quantized parameters, text/audio storage-rounded parity passed')
# Converter must reject unsupported variants rather than silently misinterpret them.
c.audio_config.enable_fusion = True
(out / 'config.json').write_text(c.to_json_string())
p = subprocess.run([sys.executable, str(ROOT / 'convert_hf_to_gguf.py'), str(out), str(out / 'invalid.gguf')], capture_output=True, text=True)
assert p.returncode != 0 and 'fusion' in p.stderr.lower(), p.stderr
assert not (out / 'invalid.gguf').exists()
# The native loader must reject malformed/truncated containers and wrong shapes.
blob = (out / 'model.gguf').read_bytes()
corruptions = [('truncated', blob[:-32], 'truncated'),
               ('architecture', blob.replace(struct.pack('<Q', 4) + b'clap', struct.pack('<Q', 4) + b'nope', 1), 'architecture')]
name = b'audio.patch_embed.proj.weight'
shape_blob = bytearray(blob)
pos = blob.index(name) + len(name) + 4  # skip name and rank (uint32)
struct.pack_into('<Q', shape_blob, pos, 3)  # expected kernel width is four
corruptions.append(('shape', bytes(shape_blob), 'shape'))
for name, data, error in corruptions:
    path = out / (name + '.gguf')
    path.write_bytes(data)
    result = subprocess.run([str(build / 'clap-model'), '--model', str(path), '--text', 'hello'], capture_output=True, text=True)
    assert result.returncode == 1 and error in result.stderr.lower(), result.stderr
print('Transformers F32/F16/Q8_0 parity, malformed GGUF and unsupported-config rejection passed')
