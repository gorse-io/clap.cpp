#!/usr/bin/env python3
"""Create a deterministic tiny CLaMP3 oracle. This is synthetic test data,
not evidence of parity with the published SAAS/C2 checkpoints.
"""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import torch
from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram
from transformers import BertConfig, BertModel, XLMRobertaConfig, XLMRobertaModel

build, out = Path(sys.argv[1]), Path(sys.argv[2])
root = Path(__file__).resolve().parents[2]
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
torch.manual_seed(314159)

vocab = [
    ('<s>', 0.0), ('<pad>', 0.0), ('</s>', 0.0), ('<unk>', 0.0),
    ('<mask>', 0.0), ('▁', -3.0), ('a', -2.0), ('b', -2.1), ('c', -2.2),
    ('▁a', -0.2), ('▁ab', -0.1), ('bc', -0.3), ('é', -0.4),
    ('▁é', -0.1), ('音', -0.3), ('楽', -0.3), ('▁音', -0.1), ('楽a', -0.1),
]
def tiny_charsmap():
    """A generated Darts character map, not a downloaded tokenizer blob."""
    trie = {}
    for before, after in {'e\u0301': 'é', '\u200b': ' ', '\ufeff': ' ', '\t': ' ', '\x01': ''}.items():
        node = trie
        for byte in before.encode('utf-8'):
            node = node.setdefault(byte, {})
        node['replacement'] = after
    units, replacements = [0] * 256, bytearray()
    next_block = 1
    def encode(node, index, label):
        nonlocal next_block
        base = next_block * 256
        next_block += 1
        units.extend([0] * (base + 256 - len(units)))
        leaf = 'replacement' in node
        units[index] = ((index ^ base) << 10) | label | (int(leaf) << 8)
        if leaf:
            units[base] = 0x80000000 | len(replacements)
            replacements.extend(node['replacement'].encode('utf-8') + b'\0')
        for key, child in node.items():
            if isinstance(key, int): encode(child, base ^ key, key)
    encode(trie, 0, 0)
    return struct.pack('<I', len(units) * 4) + struct.pack('<' + 'I' * len(units), *units) + replacements

tok = Tokenizer(Unigram(vocab, unk_id=3))
tok.normalizer = normalizers.Precompiled(tiny_charsmap())
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Metaspace(replacement='▁', prepend_scheme='always'),
])
tok.post_processor = processors.TemplateProcessing(
    single='<s> $A </s>', special_tokens=[('<s>', 0), ('</s>', 2)])
tok.save(str(out / 'tokenizer.json'))

hidden, projection, chunk = 8, 6, 128
text_config = XLMRobertaConfig(
    vocab_size=len(vocab), hidden_size=hidden, num_hidden_layers=2,
    num_attention_heads=2, intermediate_size=19, max_position_embeddings=132,
    type_vocab_size=1, pad_token_id=1, bos_token_id=0, eos_token_id=2,
    hidden_act='gelu', layer_norm_eps=1e-5,
    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
audio_config = BertConfig(
    vocab_size=1, hidden_size=hidden, num_hidden_layers=2,
    num_attention_heads=2, intermediate_size=21, max_position_embeddings=chunk,
    type_vocab_size=2, pad_token_id=0, hidden_act='gelu', layer_norm_eps=1e-12,
    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)

class TinyClamp3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = XLMRobertaModel(text_config)
        self.text_proj = torch.nn.Linear(hidden, projection)
        self.audio_model = BertModel(audio_config)
        self.audio_proj = torch.nn.Linear(hidden, projection)
        # The real checkpoint also contains unsupported symbolic tensors.
        self.symbolic_proj = torch.nn.Linear(hidden, projection)

model = TinyClamp3().eval()
torch.save({'epoch': 7, 'min_eval_loss': 0.25, 'model': model.state_dict()}, out / 'tiny-clamp3.pth')
config = {
    'model_type': 'clamp3', 'variant': 'synthetic-test',
    'projection_dim': projection, 'chunk_length': chunk,
    'audio_feature_dim': hidden,
    'text_config': text_config.to_dict(), 'audio_config': audio_config.to_dict(),
}
(out / 'clamp3_config.json').write_text(json.dumps(config))

def chunks(data, pad):
    n = len(data)
    assert n
    values = [data[i:i + chunk] for i in range(0, n, chunk)]
    values[-1] = data[-chunk:]
    result = []
    for value in values:
        length = len(value)
        if value.ndim == 1:
            padded = torch.full((chunk,), pad, dtype=value.dtype)
        else:
            padded = torch.zeros((chunk, value.shape[1]), dtype=value.dtype)
        padded[:length] = value
        mask = torch.zeros(chunk, dtype=torch.long); mask[:length] = 1
        result.append((padded, mask))
    full, remain = divmod(n, chunk)
    weights = [chunk] * full + ([remain] if remain else [])
    return result, weights

@torch.no_grad()
def text_oracle(ids):
    data = torch.tensor(ids, dtype=torch.long)
    outputs = []
    segments, weights = chunks(data, text_config.pad_token_id)
    for segment, mask in segments:
        hidden_state = model.text_model(segment[None], attention_mask=mask[None]).last_hidden_state
        pooled = (hidden_state * mask[None, :, None]).sum(1) / mask.sum()
        outputs.append(model.text_proj(pooled)[0])
    weights = torch.tensor(weights, dtype=torch.float32)
    return (torch.stack(outputs) * weights[:, None]).sum(0) / weights.sum()

@torch.no_grad()
def audio_oracle(features):
    data = torch.cat((torch.zeros(1, hidden), features, torch.zeros(1, hidden)))
    outputs = []
    segments, weights = chunks(data, 0)
    for segment, mask in segments:
        hidden_state = model.audio_model(inputs_embeds=segment[None], attention_mask=mask[None]).last_hidden_state
        pooled = (hidden_state * mask[None, :, None]).sum(1) / mask.sum()
        outputs.append(model.audio_proj(pooled)[0])
    weights = torch.tensor(weights, dtype=torch.float32)
    return (torch.stack(outputs) * weights[:, None]).sum(0) / weights.sum()

tokenizer_texts = ['ab c', 'e\u0301  音楽a', 'a <mask> b',
                   'a\u200bb\ufeffc', '\x01ab', '각', '🫨🫩', 'a\t b']
tokenizer_cases = [{'text': text, 'ids': tok.encode(text).ids} for text in tokenizer_texts]
(out / 'tokenizer-cases.json').write_text(json.dumps(tokenizer_cases))
short_ids = tok.encode('ab c').ids
long_ids = [0] + [9 if i % 3 else 10 for i in range(129)] + [2]
long_prompt = 'ab c ' * 200
text_cases = [
    {'name': 'short text', 'ids': short_ids, 'expected': text_oracle(short_ids).tolist()},
    {'name': 'overlapping long text', 'ids': long_ids, 'expected': text_oracle(long_ids).tolist()},
    {'name': 'untruncated UTF-8 text', 'ids': tok.encode(long_prompt).ids,
     'utf8': long_prompt, 'expected': text_oracle(tok.encode(long_prompt).ids).tolist()},
]
for length in (128, 256):
    ids = [0] + [9] * (length - 2) + [2]
    text_cases.append({'name': f'exact {length}-token text', 'ids': ids,
                       'expected': text_oracle(ids).tolist()})
short_audio = torch.linspace(-0.75, 0.8, 5 * hidden).reshape(5, hidden)
long_audio = torch.sin(torch.arange(131 * hidden, dtype=torch.float32) / 17).reshape(131, hidden)
audio_cases = [
    {'name': 'short audio', 'frames': len(short_audio), 'features': short_audio.flatten().tolist(),
     'expected': audio_oracle(short_audio).tolist()},
    {'name': 'overlapping long audio', 'frames': len(long_audio), 'features': long_audio.flatten().tolist(),
     'expected': audio_oracle(long_audio).tolist()},
]
for frames in (126, 254):
    features = torch.sin(torch.arange(frames * hidden, dtype=torch.float32) / 13).reshape(frames, hidden)
    audio_cases.append({'name': f'exact {frames + 2}-frame bordered audio', 'frames': frames,
                        'features': features.flatten().tolist(), 'expected': audio_oracle(features).tolist()})
fixture = dict(projection_dim=projection, embedding_tolerance=8e-5,
               tokenizer_cases=tokenizer_cases, text_cases=text_cases, audio_cases=audio_cases)
(out / 'fixture.json').write_text(json.dumps(fixture))
np.save(out / 'short-audio.npy', short_audio.numpy()[None])

converter = root / 'convert_hf_to_gguf.py'
subprocess.run([sys.executable, str(converter), str(out / 'tiny-clamp3.pth'), str(out / 'model.gguf'),
                '--clamp3-config', str(out / 'clamp3_config.json'), '--tokenizer', str(out / 'tokenizer.json')], check=True)
subprocess.run([sys.executable, str(converter), str(out / 'tiny-clamp3.pth'), str(out / 'model-f16.gguf'),
                '--clamp3-config', str(out / 'clamp3_config.json'), '--tokenizer', str(out / 'tokenizer.json'),
                '--dtype', 'f16'], check=True)
rounded = json.loads(json.dumps(fixture)); rounded['embedding_tolerance'] = 2e-3
(out / 'fixture-f16.json').write_text(json.dumps(rounded))

# Corrupt the architecture string in-place without altering GGUF offsets.
bad = bytearray((out / 'model.gguf').read_bytes())
at = bad.find(b'clamp3')
assert at >= 0
bad[at:at + 6] = b'broken'
(out / 'bad-architecture.gguf').write_bytes(bad)
print('Synthetic HF CLaMP3 F32/F16 fixture generated')
