#!/usr/bin/env python3
"""Independent tiny VoiceCLAP oracle using official Transformers operators.

Run: python tests/voiceclap_parity.py BUILD [FIXTURES] [--helper PATH]
     python tests/voiceclap_parity.py BUILD [FIXTURES] --reference-only
Requires torch, transformers==5.6.0 (reviewed Qwen2.5 Omni source), safetensors,
numpy and jinja2. CTest passes the exact helper path; manual BUILD discovery
supports standard single-/multi-config layouts (use --helper for custom layouts).
Downloads only pinned tokenizer/config/template files, NEVER model checkpoints.
Native helper target: voiceclap-parity, linked to clap, private src include path.
F16 means rounded stored weights with F32 activations, not half activations.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

REVISION = 'ae7e1d0430d131ff1c07fdaecb813637f4c0c8b0'
TRANSFORMERS_VERSION = '5.6.0'
OFFICIAL_CLASS = 'Qwen2_5OmniThinkerForConditionalGeneration'
OFFICIAL_SOURCE_SHA256 = '7440eeda3f9053d7b9d26dee2e14e6d3dc2b0c70b8a06bfd6dadf8834cec123f'
REPO = Path(__file__).resolve().parents[1]
ASSETS = {
    'modules.json': 'modules.json',
    'config.json': 'config.json',
    'preprocessor_config.json': 'preprocessor_config.json',
    'tokenizer.json': 'tokenizer.json',
    'tokenizer_config.json': 'tokenizer_config.json',
    'additional_chat_templates/sentence_transformers.jinja':
        'additional_chat_templates_sentence_transformers.jinja',
    '1_Pooling/config.json': 'pooling_config.json',
    'config_sentence_transformers.json': 'config_sentence_transformers.json',
    'sentence_bert_config.json': 'sentence_bert_config.json',
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def assets(destination, local):
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for remote, basename in ASSETS.items():
        target = destination / remote
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            source = local / basename if local else None
            if source and source.exists():
                shutil.copyfile(source, target)
            else:
                urllib.request.urlretrieve(
                    f'https://huggingface.co/VoiceNet/voiceclap-large/resolve/{REVISION}/{remote}', target)
        hashes[remote] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def validate_oracle_identity(identity):
    expected = {'revision': REVISION, 'transformers': TRANSFORMERS_VERSION,
                'official_class': OFFICIAL_CLASS, 'official_source_sha256': OFFICIAL_SOURCE_SHA256}
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f'Unreviewed oracle {key}: expected {value!r}, got {identity.get(key)!r}')


def generate(root, local):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f'Unreviewed oracle transformers: requires {TRANSFORMERS_VERSION}, '
                         f'got {transformers.__version__}')
    from safetensors.torch import save_file
    from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    model_file = Path(inspect.getfile(Qwen2_5OmniThinkerForConditionalGeneration))
    oracle_identity = {'revision': REVISION, 'transformers': transformers.__version__,
                       'official_class': Qwen2_5OmniThinkerForConditionalGeneration.__name__,
                       'official_source_sha256': hashlib.sha256(model_file.read_bytes()).hexdigest()}
    validate_oracle_identity(oracle_identity)
    torch.set_num_threads(1)
    torch.manual_seed(41927)
    source = root / 'source'
    identity = assets(root / 'assets', local)
    shutil.copytree(root / 'assets', source, dirs_exist_ok=True)
    config = json.loads((source / 'config.json').read_text())
    config['audio_config'].update(d_model=32, encoder_attention_heads=4,
                                  encoder_ffn_dim=64, encoder_layers=2,
                                  num_hidden_layers=2, output_dim=32)
    config['text_config'].update(hidden_size=32, intermediate_size=64,
                                 num_attention_heads=4, num_key_value_heads=2,
                                 num_hidden_layers=2, max_position_embeddings=2048,
                                 max_window_layers=2, layer_types=['full_attention'] * 2,
                                 rope_scaling={'mrope_section': [1, 1, 2], 'rope_type': 'default',
                                               'type': 'default'})
    config['vision_config'].update(depth=1, hidden_size=32, embed_dim=32,
                                   intermediate_size=64, num_heads=4,
                                   out_hidden_size=32, fullatt_block_indexes=[0])
    config['dtype'] = 'float32'
    config['torch_dtype'] = 'float32'
    # Read config BEFORE overwriting the copied official file. Always start fresh.
    dump(source / 'config.json', config)
    pooling = json.loads((source / '1_Pooling/config.json').read_text())
    pooling['embedding_dimension'] = 32
    dump(source / '1_Pooling/config.json', pooling)
    thinker_config = Qwen2_5OmniThinkerConfig(**config)
    thinker_config._attn_implementation = 'eager'
    model = Qwen2_5OmniThinkerForConditionalGeneration(thinker_config).eval()
    torch.manual_seed(41927)  # independent of upstream constructor initialization order
    with torch.no_grad():
        for name, value in model.named_parameters():
            if 'visual.' in name or name == 'lm_head.weight':
                continue
            if name.endswith('weight') and value.ndim == 1:
                value.copy_(1.0 + 0.04 * torch.randn_like(value))
            else:
                value.copy_(torch.randn_like(value) * (0.11 if value.ndim > 1 else 0.025))
    weights = {name: value.detach().clone().contiguous() for name, value in model.state_dict().items()
               if name.startswith(('model.', 'audio_tower.'))}
    save_file(weights, source / 'model.safetensors')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(source / 'tokenizer.json'))
    tokenizer.chat_template = (source / 'additional_chat_templates/sentence_transformers.jinja').read_text()
    extractor = WhisperFeatureExtractor(**json.loads((source / 'preprocessor_config.json').read_text()))
    text_cases = ['A quiet cello and a bright piano.', 'Thunder, footsteps, then silence.',
                  '音乐 café — soft rain.']
    inputs = []
    for i, text in enumerate(text_cases):
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}],
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        inputs.append({'name': f'text_{i}', 'mode': 'text', 'argument': text,
                       'template': rendered, 'ids': ids})
    token_cases = []
    for text in text_cases + ['', 'café', 'cafe\u0301', "We're I'LL don't can't 0123456789 98.76",
                              '中文音乐，こんにちは！', ' a  b\t\r\n c\n\n',
                              'literal <|im_start|> <|im_end|> <|audio_bos|> markup',
                              '🙂🎵 — punctuation... !!!\n']:
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}],
            tokenize=False, add_generation_prompt=True)
        token_cases.append({'text': text, 'ids': tokenizer.encode(rendered, add_special_tokens=False),
                            'template': rendered})
    assert token_cases[4]['ids'] == token_cases[5]['ids'], 'NFC reference mismatch'
    assert '\nSummarize the above text in one word:' in token_cases[0]['template']
    # Hop boundary, odd conv/pool tail and multiple independent attention chunks.
    audio_inputs = []
    for name, count, phase in [('short', 3200, 0.2), ('fractional', 10883, 0.7),
                               ('multichunk', 64643, 1.1)]:
        t = np.arange(count, dtype=np.float64) / 16000
        pcm = (0.19 * np.sin(2 * np.pi * (193 * t + 27 * t*t) + phase)
               + 0.07 * np.cos(2 * np.pi * 1601 * t)
               + 0.03 * np.sin(2 * np.pi * 19 * t)).astype('<f4')
        pcm[-9:] += np.linspace(0.0, 0.13, 9, dtype=np.float32)
        pcm.tofile(root / f'{name}.f32')
        batch = extractor(pcm, sampling_rate=16000, return_attention_mask=True,
                          return_tensors='pt', padding='max_length')
        mel_count = int(batch.attention_mask.sum())
        mel = batch.input_features[0, :, :mel_count].contiguous()
        np.save(root / f'{name}.mel.npy', mel.T.numpy())
        mel.to(torch.float32).numpy().tofile(root / f'{name}.mel-frequency-major.f32')
        audio_inputs.append((name, mel, pcm.size))
    cases = []
    for dtype in ('f32', 'f16'):
        rounded = {name: value.half().float() if dtype == 'f16' and value.ndim > 1 else value
                   for name, value in weights.items()}
        model.load_state_dict(rounded, strict=False)
        with torch.inference_mode():
            for item in inputs:
                ids = torch.tensor([item['ids']])
                mask = torch.ones_like(ids)
                # Official mRoPE position generator; audio-only and text are 1D on all axes.
                pos, _ = model.get_rope_index(ids, attention_mask=mask)
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                               use_cache=False, output_hidden_states=True)
                hidden = output.hidden_states[-1]
                value = F.normalize(hidden[0, -1].float(), dim=0).tolist()
                del output
                cases.append(dict(item, dtype=dtype, expected=value))
            for name, mel, pcm_count in audio_inputs:
                lengths = torch.tensor([mel.shape[1]])
                aftercnn, output_lengths = model.audio_tower._get_feat_extract_output_lengths(lengths)
                encoded = model.audio_tower(mel, feature_lens=lengths,
                                             aftercnn_lens=aftercnn).last_hidden_state
                assert encoded.shape == (int(output_lengths[0]), 32)
                rendered = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': [{'type': 'audio', 'audio': 'fixture'}]}],
                    tokenize=False, add_generation_prompt=True)
                rendered = rendered.replace('<|AUDIO|>', '<|AUDIO|>' * encoded.shape[0])
                ids = torch.tensor([tokenizer.encode(rendered, add_special_tokens=False)])
                selected = ids == config['audio_token_index']
                assert int(selected.sum()) == encoded.shape[0]
                embeds = model.model.embed_tokens(ids)
                embeds[selected] = encoded
                mask = torch.ones_like(ids)
                pos, _ = model.get_rope_index(ids, attention_mask=mask, audio_seqlens=lengths)
                assert torch.equal(pos[0], torch.arange(ids.shape[1]).unsqueeze(0))
                hidden = model.model(inputs_embeds=embeds, attention_mask=mask,
                                     position_ids=pos, use_cache=False).last_hidden_state
                # Exercise the official multimodal wrapper too, including its actual
                # feature extraction, placeholder replacement and position generation.
                output = model(input_ids=ids, attention_mask=mask,
                               input_features=mel.unsqueeze(0),
                               feature_attention_mask=torch.ones((1, mel.shape[1]), dtype=torch.long),
                               output_hidden_states=True, use_cache=False)
                full_hidden = output.hidden_states[-1]
                torch.testing.assert_close(full_hidden, hidden, rtol=1e-6, atol=1e-6)
                value = F.normalize(full_hidden[0, -1].float(), dim=0).tolist()
                del output
                cases.append({'name': name, 'mode': 'audio', 'argument': str(root / f'{name}.f32'),
                              'dtype': dtype, 'template': rendered, 'ids': ids[0].tolist(),
                              'samples': pcm_count, 'mel_frames': mel.shape[1],
                              'audio_tokens': encoded.shape[0], 'expected': value})
    for dtype in ('f32', 'f16'):
        vectors = [np.array(x['expected']) for x in cases if x['dtype'] == dtype]
        assert all(np.isfinite(x).all() and abs(np.linalg.norm(x)-1) < 1e-6 for x in vectors)
        assert np.max(np.abs(vectors[0]-vectors[1])) > 1e-3, 'degenerate text oracle'
        assert np.max(np.abs(vectors[-1]-vectors[-2])) > 1e-4, 'degenerate audio oracle'
    report = {**oracle_identity, 'torch': torch.__version__,
              'assets_sha256': identity, 'seed': 41927, 'cases': cases, 'token_cases': token_cases,
              'fixture_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                  for name in ('model.safetensors', 'config.json')},
              'limitations': ['Tiny random official weights; not published-checkpoint parity.',
                              'F16 storage is tested with explicit F32 activations.']}
    dump(root / 'oracle.json', report)
    print(f'Official Transformers reference: {len(cases)} embedding cases, 3 waveform frontends', flush=True)
    return report


def helper_path(build, helper=None):
    if helper is not None:
        helper = Path(helper).resolve()
        if not helper.is_file():
            raise FileNotFoundError(f'--helper is not a file: {helper}')
        return helper
    for directory in (build, build / 'tests', build / 'bin'):
        for config in ('', 'Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel'):
            for name in ('voiceclap-parity', 'voiceclap-parity.exe'):
                candidate = directory / config / name
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(f'Build native target voiceclap-parity in {build}, '
                            'or pass --helper PATH for its exact executable/configuration')


def run_json(helper, model, mode, argument=None):
    args = [str(helper), str(model), mode]
    if argument is not None:
        args.append(str(argument))
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f'Invalid helper JSON: {completed.stdout}\n{completed.stderr}') from error


def compare(name, actual, expected, atol, metrics):
    import numpy as np
    a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == e.shape, (name, a.shape, e.shape)
    assert np.isfinite(a).all(), name
    error = float(np.max(np.abs(a-e)))
    metrics.append({'name': name, 'max_abs': error, 'tolerance': atol})
    print(f'{name}: max_abs={error:.8g} (limit {atol:g})', flush=True)
    assert error <= atol, f'{name}: {error} > {atol}'


def verify(build, root, report, helper=None):
    import numpy as np
    helper = helper_path(build, helper)
    metrics = []
    negative = None
    rejected = None
    for i, case in enumerate(report['token_cases']):
        actual = run_json(helper, root / 'source/tokenizer.json', 'tokenize', case['text'])
        assert actual['ids'] == case['ids'], f'tokenizer case {i}: {actual["ids"]} != {case["ids"]}'
    print(f'Tokenizer: {len(report["token_cases"])} exact official ID sequences', flush=True)
    for dtype in ('f32', 'f16'):
        model = root / f'voiceclap-{dtype}.gguf'
        subprocess.run([sys.executable, str(REPO / 'convert_hf_to_gguf.py'), str(root / 'source'),
                        str(model), '--dtype', dtype], check=True)
        for case in report['cases']:
            if case['dtype'] != dtype:
                continue
            actual = run_json(helper, model, case['mode'], case['argument'])
            compare(f'{dtype}/{case["mode"]}/{case["name"]}', actual['values'], case['expected'],
                    4e-4, metrics)
        if dtype == 'f32':
            for name in ('short', 'fractional', 'multichunk'):
                actual = run_json(helper, model, 'frontend', root / f'{name}.f32')
                expected = np.load(root / f'{name}.mel.npy')
                assert (actual['frames'], actual['dimension']) == expected.shape
                compare(f'frontend/{name}', np.asarray(actual['values']).reshape(expected.shape),
                        expected, 3e-4, metrics)
            negative = run_json(helper, model, 'invalid', root / 'short.f32')
            assert all(negative[key] != 0 for key in ('empty', 'crop', 'nan')), negative
            # Bypass the converter: a native loader must reject unsupported GGUF configs itself.
            import re
            data = model.read_bytes()
            bad_data, replacements = re.subn(br'("hidden_act"\s*:\s*)"silu"',
                                             br'\1"gelu"', data, count=1)
            assert replacements == 1 and len(bad_data) == len(data)
            unsupported = root / 'voiceclap-unsupported-activation.gguf'
            unsupported.write_bytes(bad_data)
            rejected = run_json(helper, unsupported, 'reject')
            assert rejected['load_status'] != 0, 'native loader accepted unsupported text activation'
    dump(root / 'native-results.json', {'metrics': metrics, 'invalid': negative,
                                       'token_cases': len(report['token_cases']), 'unsupported': rejected})
    print(f'PASS: {len(metrics)} numerical comparisons; empty/crop/NaN and unsupported config rejected', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('fixtures', type=Path, nargs='?')
    parser.add_argument('--helper', type=Path, help='Exact native voiceclap-parity executable path')
    parser.add_argument('--assets', type=Path, help='Local pinned official assets (no checkpoints)')
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--reuse-reference', action='store_true', help='Use existing oracle.json')
    args = parser.parse_args()
    build = args.build.resolve()
    root = (args.fixtures or build / 'voiceclap-fixtures').resolve()
    helper = None if args.reference_only else helper_path(build, args.helper)
    root.mkdir(parents=True, exist_ok=True)
    if args.reuse_reference:
        report = json.loads((root / 'oracle.json').read_text())
        validate_oracle_identity(report)
        for name, expected in report['fixture_sha256'].items():
            actual = hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest()
            assert actual == expected, f'fixture changed since reference generation: {name}'
    else:
        report = generate(root, args.assets)
    if not args.reference_only:
        verify(build, root, report, helper=helper)


if __name__ == '__main__':
    main()
