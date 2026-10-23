"""Small valid CLAP/MERT GGUFs, generated with stdlib only (no downloads)."""
import json
import math
from pathlib import Path
import struct


def write_gguf(path, metadata, tensors):
    def string(s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += string(key) + struct.pack('<I', 8) + string(value)
    payload = bytearray()
    for name, shape in tensors.items():
        header += (string(name) + struct.pack('<I', len(shape)) +
                   struct.pack('<' + 'Q' * len(shape), *shape) +
                   struct.pack('<IQ', 0, len(payload)))
        count = math.prod(shape)
        # Unit norm scales/running variance, zero norm bias, deterministic dense
        # weights. Round to F32 in the actual file rather than mocking inference.
        norm = 'norm' in name.lower() or 'LayerNorm' in name
        vals = [(1.0 if name.endswith('weight') or name.endswith('running_var') else 0.0)
                if norm else 0.12 * math.sin(i * 0.73 + sum(name.encode()) * 0.17)
                for i in range(count)]
        payload.extend(struct.pack('<' + 'f' * count, *vals))
        payload.extend(b'\0' * (-len(payload) % 32))
    Path(path).write_bytes(header + b'\0' * (-len(header) % 32) + payload)


def clap_fixture(path):
    shapes = {}
    def norm(p, d):
        shapes[p + '.weight'] = (d,)
        shapes[p + '.bias'] = (d,)
    def linear(p, i, o, bias=True):
        shapes[p + '.weight'] = (i, o)
        if bias:
            shapes[p + '.bias'] = (o,)
    a = dict(enable_fusion=False, depths=[1]*4, num_attention_heads=[1]*4,
             patch_size=4, patch_stride=[4, 4], patch_embed_input_channels=1,
             qkv_bias=True, enable_patch_layer_norm=True, flatten_patch_embeds=True,
             hidden_act='gelu', projection_hidden_act='relu', spec_size=256,
             num_mel_bins=64, window_size=2, patch_embeds_hidden_size=2,
             hidden_size=16, mlp_ratio=2.0, projection_dim=3, layer_norm_eps=1e-5)
    t = dict(hidden_act='gelu', projection_hidden_act='relu', hidden_size=4,
             num_attention_heads=1, num_hidden_layers=1, type_vocab_size=1,
             vocab_size=261, max_position_embeddings=514, intermediate_size=8,
             projection_dim=3, bos_token_id=0, eos_token_id=2, pad_token_id=1,
             layer_norm_eps=1e-5)
    shapes['audio.patch_embed.proj.weight'] = (4, 4, 1, 2)
    shapes['audio.patch_embed.proj.bias'] = (2,)
    norm('audio.patch_embed.norm', 2)
    norm('audio.batch_norm', 64)
    shapes['audio.batch_norm.running_mean'] = (64,)
    shapes['audio.batch_norm.running_var'] = (64,)
    d = 2
    for stage in range(4):
        p = f'audio.layers.{stage}.blocks.0'
        norm(p + '.layernorm_before', d)
        norm(p + '.layernorm_after', d)
        for part in ('query', 'key', 'value'):
            linear(p + '.attention.self.' + part, d, d)
        shapes[p + '.attention.self.rel_bias'] = (1, 9)
        linear(p + '.attention.output.dense', d, d)
        linear(p + '.intermediate.dense', d, 2*d)
        linear(p + '.output.dense', 2*d, d)
        if stage < 3:
            p = f'audio.layers.{stage}.downsample'
            norm(p + '.norm', 4*d)
            linear(p + '.reduction', 4*d, 2*d, False)
        d *= 2
    norm('audio.norm', 16)
    for name, n in [('word', 261), ('position', 514), ('token_type', 1)]:
        shapes[f'text.embeddings.{name}_embeddings.weight'] = (4, n)
    norm('text.embeddings.LayerNorm', 4)
    p = 'text.encoder.layer.0'
    for part in ('query', 'key', 'value'):
        linear(p + '.attention.self.' + part, 4, 4)
    linear(p + '.attention.output.dense', 4, 4)
    norm(p + '.attention.output.LayerNorm', 4)
    linear(p + '.intermediate.dense', 4, 8)
    linear(p + '.output.dense', 8, 4)
    norm(p + '.output.LayerNorm', 4)
    linear('text.pooler.dense', 4, 4)
    for branch, width in [('text', 4), ('audio', 16)]:
        linear(branch + '_proj.linear1', width, 3)
        linear(branch + '_proj.linear2', 3, 3)
    bytechars = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    extra = 256
    mapping = []
    for b in range(256):
        mapping.append(chr(b if b in bytechars else extra))
        if b not in bytechars:
            extra += 1
    vocab = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '<mask>': 4}
    vocab.update({ch: i + 5 for i, ch in enumerate(mapping)})
    tok = dict(model=dict(type='BPE', vocab=vocab, merges=[]))
    config = dict(model_type='clap', projection_dim=3, audio_config=a, text_config=t)
    write_gguf(path, {'general.architecture': 'clap', 'clap.config': json.dumps(config),
                      'clap.tokenizer': json.dumps(tok)}, shapes)


def mert_fixture(path):
    # One aggressive stride keeps a >=1 second waveform tiny after the frontend.
    c = dict(model_type='mert_model', sample_rate=24000, conv_bias=False,
             feat_extract_norm='group', hidden_act='gelu', feat_extract_activation='gelu',
             hidden_size=4, intermediate_size=8, num_hidden_layers=1,
             num_attention_heads=1, num_conv_pos_embeddings=4,
             num_conv_pos_embedding_groups=1, conv_dim=[2], conv_kernel=[4],
             conv_stride=[320], layer_norm_eps=1e-5)
    s = {'frontend.0.conv.weight': (4, 1, 2), 'position.weight': (4, 4, 4),
         'position.bias': (4,)}
    def norm(p, d):
        s[p + '.weight'] = (d,)
        s[p + '.bias'] = (d,)
    def linear(p, i, o):
        s[p + '.weight'] = (i, o)
        s[p + '.bias'] = (o,)
    norm('frontend.0.layer_norm', 2)
    norm('projection.layer_norm', 2)
    linear('projection.projection', 2, 4)
    norm('encoder.layer_norm', 4)
    for part in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
        linear('layers.0.attention.' + part, 4, 4)
    norm('layers.0.layer_norm', 4)
    norm('layers.0.final_layer_norm', 4)
    linear('layers.0.ffn.in', 4, 8)
    linear('layers.0.ffn.out', 8, 4)
    write_gguf(path, {'general.architecture': 'mert', 'mert.config': json.dumps(c)}, s)


if __name__ == '__main__':
    import sys
    clap_fixture(sys.argv[1])
