"""Stdlib-only VoiceCLAP conversion contracts; no checkpoints downloaded."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversion import get_model_architecture, get_model_class


class VoiceClapConversionTest(unittest.TestCase):
    def test_lazy_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'config.json').write_text(json.dumps({'model_type': 'qwen2_5_omni_thinker'}))
            self.assertEqual(get_model_architecture(source), 'voiceclap')
            self.assertEqual(get_model_class('voiceclap').__name__, 'VoiceClapModel')

    def test_compact_names(self):
        module = importlib.import_module('conversion.voiceclap')
        self.assertEqual(module.tensor_name('model.layers.27.post_attention_layernorm.weight'),
                         'text.layers.27.ffn_norm.weight')
        self.assertEqual(module.tensor_name('audio_tower.layers.31.self_attn_layer_norm.weight'),
                         'audio.layers.31.attn_norm.weight')
        self.assertEqual(module.tensor_name('audio_tower.layers.0.self_attn.out_proj.weight'),
                         'audio.layers.0.attn.out_proj.weight')
        self.assertIsNone(module.tensor_name('visual.blocks.0.weight'))
        self.assertIsNone(module.tensor_name('lm_head.weight'))
        with self.assertRaises(ValueError):
            module.tensor_name('adapter.unknown.weight')

    def test_actual_streamed_f32_f16_bf16(self):
        from conversion.base import write_gguf_streaming
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'weights'
            # Independent BF16 values (1, -2, .5, 3), then F32 vector.
            raw.write_bytes(struct.pack('<4H', 0x3f80, 0xc000, 0x3f00, 0x4040) + struct.pack('<2f', .25, -.75))
            tensors = [dict(name='text.matrix', shape=[2, 2], dtype='BF16', path=raw, offset=0),
                       dict(name='text.bias', shape=[2], dtype='F32', path=raw, offset=8)]
            for dtype, matrix_type, fmt in [('f32', 0, '<4f'), ('f16', 1, '<4e')]:
                target = root / (dtype + '.gguf')
                write_gguf_streaming(target, tensors, {'general.architecture': 'voiceclap'}, dtype)
                metadata, entries, start = read_gguf(target)
                data = target.read_bytes()
                self.assertEqual(metadata['general.architecture'], 'voiceclap')
                self.assertEqual(entries[0][1:3], ([2, 2], matrix_type))
                offset = start + entries[0][3]
                self.assertEqual(struct.unpack_from(fmt, data, offset), (1., -2., .5, 3.))
                self.assertEqual(entries[1][2], 0)
                self.assertEqual(struct.unpack_from('<2f', data, start + entries[1][3]), (.25, -.75))

    @unittest.skipUnless(importlib.util.find_spec('tokenizers'), 'optional tokenizers package not installed')
    def test_full_converter_f32_f16(self):
        module = importlib.import_module('conversion.voiceclap')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            make_source(source)
            for dtype in ('f32', 'f16'):
                output = source / (dtype + '.gguf')
                get_model_class('voiceclap')(source, output, dtype).write()
                metadata, tensors, start = read_gguf(output)
                self.assertEqual(metadata['general.architecture'], 'voiceclap')
                self.assertEqual(json.loads(metadata['voiceclap.config']), CONFIG)
                self.assertEqual(metadata['voiceclap.chat_template'], TEMPLATE)
                self.assertTrue(json.loads(metadata['voiceclap.audio_prefix']))
                self.assertTrue(json.loads(metadata['voiceclap.audio_suffix']))
                self.assertEqual(len(tensors), 38)  # 12 text/layer + 15 audio/layer + 11 base
                self.assertTrue(all(len(t[0].encode()) < 64 for t in tensors))
                self.assertFalse(any(t[0].startswith(('visual.', 'lm_head.')) for t in tensors))
                for name, shape, typ, offset in tensors:
                    self.assertEqual(typ, int(dtype == 'f16' and len(shape) > 1))
                    value, = struct.unpack_from('<e' if typ else '<f', output.read_bytes(), start + offset)
                    self.assertEqual(value, .5)

    def test_strict_sidecars_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            make_source(source)
            cls = get_model_class('voiceclap')
            for filename, replacement in [('1_Pooling/config.json', {'embedding_dimension': 4, 'pooling_mode': 'mean'}),
                                           ('modules.json', []),
                                           ('preprocessor_config.json', {'sampling_rate': 24000}),
                                           ('sentence_bert_config.json', {}),
                                           ('config_sentence_transformers.json', {})]:
                path = source / filename
                saved = path.read_text()
                path.write_text(json.dumps(replacement))
                with self.assertRaises(ValueError, msg=filename):
                    cls(source, source / 'out.gguf').write()
                path.write_text(saved)
            path = source / 'additional_chat_templates/sentence_transformers.jinja'
            path.write_text('{{ messages }}')
            with self.assertRaises(ValueError):
                cls(source, source / 'out.gguf').write()
            path.write_text(TEMPLATE)
            # Missing required tensor is rejected before producing a target.
            write_checkpoint(source / 'model.safetensors', {'model.embed_tokens.weight': [8, 4]})
            with self.assertRaises(ValueError):
                cls(source, source / 'out.gguf').write()
            self.assertFalse((source / 'out.gguf').exists())

    @unittest.skipUnless(importlib.util.find_spec('tokenizers'), 'optional tokenizers package not installed')
    def test_sharded_and_malformed_safetensors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            shapes = make_source(source)
            items = list(shapes.items())
            (source / 'model.safetensors').unlink()
            mapping = {}
            for i, subset in enumerate((items[::2], items[1::2])):
                name = 'model-' + str(i) + '.safetensors'
                write_checkpoint(source / name, dict(subset))
                mapping.update({key: name for key, _ in subset})
            index = source / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': mapping}))
            get_model_class('voiceclap')(source, source / 'out.gguf').write()
            self.assertTrue((source / 'out.gguf').exists())
            mapping['model.norm.weight'] = '../escape.safetensors'
            index.write_text(json.dumps({'weight_map': mapping}))
            with self.assertRaises(ValueError):
                get_model_class('voiceclap')(source, source / 'bad.gguf').write()

    def test_invalid_tokenizer_policy(self):
        from conversion.voiceclap import _validate_tokenizer
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            make_source(source)
            tokenizer = json.loads((source / 'tokenizer.json').read_text(encoding='utf-8'))
            tc = json.loads((source / 'tokenizer_config.json').read_text(encoding='utf-8'))
            for changes in [{'normalizer': None}, {'model': {'type': 'WordPiece'}},
                            {'pre_tokenizer': {'type': 'Whitespace'}}, {'added_tokens': []}]:
                changed = copy.deepcopy(tokenizer)
                changed.update(changes)
                with self.assertRaises(ValueError):
                    _validate_tokenizer(changed, tc)

    def test_malformed_safetensors_headers(self):
        from conversion.voiceclap import _safetensors
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'model.safetensors'
            for header in [b'{"x":{"shape":[2],"dtype":"I64","data_offsets":[0,4]}}',
                           b'{"x":{"shape":[2],"dtype":"F32","data_offsets":[0,4]}}',
                           b'{"x":{"shape":[1],"dtype":"F32","data_offsets":[1,5]}}',
                           b'{"x":{},"x":{}}']:
                source.write_bytes(struct.pack('<Q',len(header)) + header + b'\0'*8)
                with self.assertRaises(ValueError):
                    _safetensors(source)

    def test_stream_float_storage_and_chunks(self):
        from conversion.base import write_gguf_streaming
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.write_bytes(struct.pack('<f', .25) * 65537)
            descriptor = dict(name='matrix', shape=[1,65537], dtype='F32', path=raw, offset=0)
            output = root / 'out.gguf'
            write_gguf_streaming(output, [descriptor], {'general.architecture':'voiceclap'}, 'f16')
            _, entries, start = read_gguf(output)
            self.assertEqual(entries[0][1], [65537,1])
            self.assertEqual(output.read_bytes()[start:start+65537*2], struct.pack('<e',.25)*65537)
            raw.write_bytes(struct.pack('<3e',1,-2,.25))
            descriptor.update(shape=[3], dtype='F16')
            write_gguf_streaming(output, [descriptor], {'general.architecture':'voiceclap'}, 'f32')
            _, _, start = read_gguf(output)
            self.assertEqual(struct.unpack_from('<3f',output.read_bytes(),start),(1,-2,.25))

    def test_stream_atomic_failure(self):
        from conversion.base import write_gguf_streaming
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, target = root / 'raw', root / 'output'
            raw.write_bytes(b'\0\0')
            target.write_bytes(b'original')
            with self.assertRaises(ValueError):
                write_gguf_streaming(target, [dict(name='x', shape=[8], dtype='F32', path=raw, offset=0)],
                                     {'general.architecture': 'voiceclap'}, 'f32')
            self.assertEqual(target.read_bytes(), b'original')
            self.assertEqual(set(root.iterdir()), {raw, target})


class VoiceClapParityRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            'voiceclap_parity', Path(__file__).with_name('voiceclap_parity.py'))
        assert spec is not None and spec.loader is not None
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_windows_helper_layouts(self):
        for layout in ('tests/voiceclap-parity.exe', 'tests/Release/voiceclap-parity.exe',
                       'tests/Debug/voiceclap-parity.exe', 'tests/Release/voiceclap-parity'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                build = Path(directory)
                helper = build / layout
                helper.parent.mkdir(parents=True, exist_ok=True)
                helper.touch()
                self.assertEqual(self.runner.helper_path(build), helper)

    def test_explicit_helper_with_spaces(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix='voiceclap build ') as directory:
            build = Path(directory)
            helper = build / 'custom output' / 'native helper.exe'
            helper.parent.mkdir()
            helper.touch()
            self.assertEqual(self.runner.helper_path(build, helper), helper.resolve())
            with patch.object(sys, 'argv', ['voiceclap_parity.py', str(build), '--helper',
                                            str(helper)]), \
                    patch.object(self.runner, 'generate', return_value={}), \
                    patch.object(self.runner, 'verify') as verify:
                self.runner.main()
            self.assertEqual(verify.call_args.kwargs['helper'], helper.resolve())

    def test_missing_helper_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, '--helper'):
                self.runner.helper_path(build)
            (build / 'voiceclap-parity').touch()
            with self.assertRaisesRegex(FileNotFoundError, 'missing.exe'):
                self.runner.helper_path(build, build / 'missing.exe')

    def test_reviewed_oracle_identity(self):
        identity = {'revision': self.runner.REVISION, 'transformers': '5.6.0',
                    'official_class': 'Qwen2_5OmniThinkerForConditionalGeneration',
                    'official_source_sha256':
                        '7440eeda3f9053d7b9d26dee2e14e6d3dc2b0c70b8a06bfd6dadf8834cec123f'}
        self.runner.validate_oracle_identity(identity)
        for key in identity:
            for value in ('unreviewed', None):
                changed = dict(identity)
                if value is None:
                    del changed[key]
                else:
                    changed[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.runner.validate_oracle_identity(changed)

    def test_reused_oracle_rejects_unreviewed_identity(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'oracle.json').write_text(json.dumps(
                {'revision': self.runner.REVISION, 'transformers': '5.7.0',
                 'fixture_sha256': {}}))
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('voiceclap_parity.py')),
                                     str(root), str(root), '--reference-only', '--reuse-reference'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('transformers', result.stderr)


def read_gguf(path):
    data = path.read_bytes()
    assert data[:4] == b'GGUF'
    version, count, pairs = struct.unpack_from('<IQQ', data, 4)
    assert version == 3
    cursor = 24
    def string():
        nonlocal cursor
        length, = struct.unpack_from('<Q', data, cursor)
        cursor += 8
        value = data[cursor:cursor + length].decode('utf-8')
        cursor += length
        return value
    metadata = {}
    for _ in range(pairs):
        key = string()
        typ, = struct.unpack_from('<I', data, cursor)
        cursor += 4
        assert typ == 8
        metadata[key] = string()
    entries = []
    for _ in range(count):
        name = string()
        ndim, = struct.unpack_from('<I', data, cursor)
        cursor += 4
        shape = list(struct.unpack_from('<' + 'Q' * ndim, data, cursor))
        cursor += 8 * ndim
        typ, offset = struct.unpack_from('<IQ', data, cursor)
        cursor += 12
        entries.append((name, shape, typ, offset))
    return metadata, entries, (cursor + 31) // 32 * 32


CONFIG = {'_attn_implementation_autoset': True, 'architectures': ['Qwen2_5OmniThinkerForConditionalGeneration'], 'audio_config': {'_attn_implementation_autoset': True, 'activation_dropout': 0.0, 'activation_function': 'gelu', 'attention_dropout': 0.0, 'd_model': 4, 'dropout': 0.0, 'encoder_attention_heads': 2, 'encoder_ffn_dim': 8, 'encoder_layerdrop': 0.0, 'encoder_layers': 1, 'init_std': 0.02, 'initializer_range': 0.02, 'max_source_positions': 1500, 'model_type': 'qwen2_5_omni_audio_encoder', 'n_window': 100, 'num_hidden_layers': 1, 'num_mel_bins': 128, 'output_dim': 4, 'scale_embedding': False, 'tf_legacy_loss': False, 'use_bfloat16': False}, 'audio_end_token_id': 151648, 'audio_start_token_id': 151647, 'audio_token_index': 151646, 'bos_token_id': 151644, 'dtype': 'bfloat16', 'eos_token_id': 151645, 'ignore_index': -100, 'image_token_index': 151655, 'init_std': 0.02, 'initializer_range': 0.02, 'model_type': 'qwen2_5_omni_thinker', 'pad_token_id': 151643, 'position_id_per_seconds': 25, 'seconds_per_chunk': 2, 'text_config': {'attention_dropout': 0.0, 'hidden_act': 'silu', 'hidden_size': 4, 'init_std': 0.02, 'initializer_range': 0.02, 'intermediate_size': 8, 'layer_types': ['full_attention'], 'max_position_embeddings': 32768, 'max_window_layers': 28, 'model_type': 'qwen2_5_omni_text', 'num_attention_heads': 2, 'num_hidden_layers': 1, 'num_key_value_heads': 1, 'rms_norm_eps': 1e-06, 'rope_scaling': {'mrope_section': [16, 24, 24], 'rope_type': 'default', 'type': 'default'}, 'rope_theta': 1000000.0, 'sliding_window': None, 'use_cache': True, 'use_sliding_window': False, 'vocab_size': 128}, 'transformers_version': '4.57.0.dev0', 'user_token_id': 872, 'video_token_index': 151656, 'vision_config': {'_attn_implementation_autoset': True, 'depth': 32, 'embed_dim': 1280, 'fullatt_block_indexes': [7, 15, 23, 31], 'hidden_act': 'silu', 'hidden_size': 1280, 'in_channels': 3, 'in_chans': 3, 'init_std': 0.02, 'initializer_range': 0.02, 'intermediate_size': 3420, 'model_type': 'qwen2_5_omni_vision_encoder', 'num_heads': 16, 'out_hidden_size': 3584, 'patch_size': 14, 'spatial_merge_size': 2, 'spatial_patch_size': 14, 'temporal_patch_size': 2, 'tf_legacy_loss': False, 'tokens_per_second': 25, 'use_bfloat16': False, 'window_size': 112}, 'vision_end_token_id': 151653, 'vision_start_token_id': 151652, 'vision_token_id': 151654}
TEMPLATE = "{%- set audio_count = namespace(value=0) -%}\n{%- set image_count = namespace(value=0) -%}\n{%- set video_count = namespace(value=0) -%}\n<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {#- skip: the fixed system prompt above was already emitted. Any input system\n            message is only present to silence Qwen2.5-Omni's default-prompt warning. -#}\n    {%- else -%}\n        <|im_start|>{{ message['role'] }}\n{% if message['content'] is string -%}\n        {{- message['content'] -}}<|im_end|>\n{% else -%}\n        {%- set seen = namespace(image=false, audio=false, video=false) -%}\n        {%- for content in message['content'] -%}\n            {%- if content['type'] == 'image' or 'image' in content or 'image_url' in content -%}\n                {%- set image_count.value = image_count.value + 1 -%}\n                {%- set seen.image = true -%}\n                {%- if add_vision_id -%}Picture {{ image_count.value }}: {% endif -%}\n                <|vision_bos|><|IMAGE|><|vision_eos|>\n            {%- elif content['type'] == 'audio' or 'audio' in content or 'audio_url' in content -%}\n                {%- set audio_count.value = audio_count.value + 1 -%}\n                {%- set seen.audio = true -%}\n                {%- if add_audio_id -%}Audio {{ audio_count.value }}: {% endif -%}\n                <|audio_bos|><|AUDIO|><|audio_eos|>\n            {%- elif content['type'] == 'video' or 'video' in content -%}\n                {%- set video_count.value = video_count.value + 1 -%}\n                {%- set seen.video = true -%}\n                {%- if add_vision_id -%}Video {{ video_count.value }}: {% endif -%}\n                <|vision_bos|><|VIDEO|><|vision_eos|>\n            {%- elif 'text' in content -%}\n                {{- content['text'] -}}\n            {%- endif -%}\n        {%- endfor -%}\n        {%- if seen.image -%}\n{{ '\\n' }}Summarize the above image in one word:\n        {%- elif seen.video -%}\n{{ '\\n' }}Summarize the above video in one word:\n        {%- elif seen.audio -%}\n{{ '\\n' }}Summarize the above audio in one word:\n        {%- else -%}\n{{ '\\n' }}Summarize the above text in one word:\n        {%- endif -%}\n        <|im_end|>\n{% endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    <|im_start|>assistant\n{% endif -%}\n"
ASSETS = {'modules.json': '[\n  {\n    "idx": 0,\n    "name": "0",\n    "path": "",\n    "type": "sentence_transformers.base.modules.transformer.Transformer"\n  },\n  {\n    "idx": 1,\n    "name": "1",\n    "path": "1_Pooling",\n    "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"\n  },\n  {\n    "idx": 2,\n    "name": "2",\n    "path": "2_Normalize",\n    "type": "sentence_transformers.sentence_transformer.modules.normalize.Normalize"\n  }\n]', 'preprocessor_config.json': '{\n  "chunk_length": 300,\n  "dither": 0.0,\n  "feature_extractor_type": "WhisperFeatureExtractor",\n  "feature_size": 128,\n  "hop_length": 160,\n  "image_mean": [\n    0.48145466,\n    0.4578275,\n    0.40821073\n  ],\n  "image_processor_type": "Qwen2VLImageProcessor",\n  "image_std": [\n    0.26862954,\n    0.26130258,\n    0.27577711\n  ],\n  "max_pixels": 12845056,\n  "merge_size": 2,\n  "min_pixels": 3136,\n  "n_fft": 400,\n  "n_samples": 4800000,\n  "nb_max_frames": 30000,\n  "padding_side": "right",\n  "padding_value": 0.0,\n  "patch_size": 14,\n  "processor_class": "Qwen2_5OmniProcessor",\n  "return_attention_mask": true,\n  "sampling_rate": 16000,\n  "temporal_patch_size": 2\n}\n', 'sentence_bert_config.json': '{\n    "transformer_task": "any-to-any",\n    "modality_config": {\n        "text": {\n            "method": "forward",\n            "method_output_name": [\n                "hidden_states",\n                -1\n            ]\n        },\n        "image": {\n            "method": "forward",\n            "method_output_name": [\n                "hidden_states",\n                -1\n            ]\n        },\n        "audio": {\n            "method": "forward",\n            "method_output_name": [\n                "hidden_states",\n                -1\n            ]\n        },\n        "video": {\n            "method": "forward",\n            "method_output_name": [\n                "hidden_states",\n                -1\n            ]\n        },\n        "message": {\n            "method": "forward",\n            "method_output_name": [\n                "hidden_states",\n                -1\n            ],\n            "format": "structured"\n        }\n    },\n    "module_output_name": "token_embeddings",\n    "processing_kwargs": {\n        "chat_template": {\n            "chat_template": "sentence_transformers",\n            "add_generation_prompt": true\n        }\n    }\n}', 'config_sentence_transformers.json': '{\n  "__version__": {\n    "pytorch": "2.10.0+cu128",\n    "sentence_transformers": "5.4.0",\n    "transformers": "5.6.0"\n  },\n  "default_prompt_name": "default",\n  "model_type": "SentenceTransformer",\n  "prompts": {\n    "default": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."\n  },\n  "similarity_fn_name": "cosine"\n}\n', 'tokenizer_config.json': '{\n  "add_prefix_space": false,\n  "added_tokens_decoder": {\n    "151643": {\n      "content": "<|endoftext|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151644": {\n      "content": "<|im_start|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151645": {\n      "content": "<|im_end|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151646": {\n      "content": "<|AUDIO|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151647": {\n      "content": "<|audio_bos|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151648": {\n      "content": "<|audio_eos|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151649": {\n      "content": "<|box_end|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151650": {\n      "content": "<|quad_start|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151651": {\n      "content": "<|quad_end|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151652": {\n      "content": "<|vision_bos|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151653": {\n      "content": "<|vision_eos|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151654": {\n      "content": "<|vision_pad|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151655": {\n      "content": "<|IMAGE|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151656": {\n      "content": "<|VIDEO|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": true\n    },\n    "151657": {\n      "content": "<tool_call>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": false\n    },\n    "151658": {\n      "content": "</tool_call>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": false\n    },\n    "151659": {\n      "content": "<|fim_prefix|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": false\n    },\n    "151660": {\n      "content": "<|fim_middle|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": false\n    },\n    "151661": {\n      "content": "<|fim_suffix|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": false\n    },\n    "151662": {\n      "content": "<|fim_pad|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": false\n    },\n    "151663": {\n      "content": "<|repo_name|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": false\n    },\n    "151664": {\n      "content": "<|file_sep|>",\n      "lstrip": false,\n      "normalized": false,\n      "rstrip": false,\n      "single_word": false,\n      "special": false\n    }\n  },\n  "additional_special_tokens": [\n    "<|im_start|>",\n    "<|im_end|>",\n    "<|AUDIO|>",\n    "<|audio_bos|>",\n    "<|audio_eos|>",\n    "<|box_end|>",\n    "<|quad_start|>",\n    "<|quad_end|>",\n    "<|vision_bos|>",\n    "<|vision_eos|>",\n    "<|vision_pad|>",\n    "<|IMAGE|>",\n    "<|VIDEO|>"\n  ],\n  "bos_token": null,\n  "chat_template": "{% set audio_count = namespace(value=0) %}{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message[\'role\'] != \'system\' %}<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n{% endif %}<|im_start|>{{ message[\'role\'] }}\\n{% if message[\'content\'] is string %}{{ message[\'content\'] }}<|im_end|>\\n{% else %}{% for content in message[\'content\'] %}{% if content[\'type\'] == \'image\' or \'image\' in content or \'image_url\' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_bos|><|IMAGE|><|vision_eos|>{% elif content[\'type\'] == \'audio\' or \'audio\' in content or \'audio_url\' in content %}{% set audio_count.value = audio_count.value + 1 %}{% if add_audio_id %}Audio {{ audio_count.value }}: {% endif %}<|audio_bos|><|AUDIO|><|audio_eos|>{% elif content[\'type\'] == \'video\' or \'video\' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_bos|><|VIDEO|><|vision_eos|>{% elif \'text\' in content %}{{ content[\'text\'] }}{% endif %}{% endfor %}<|im_end|>\\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}",\n  "clean_up_tokenization_spaces": false,\n  "eos_token": "<|im_end|>",\n  "errors": "replace",\n  "extra_special_tokens": {\n    "image_token": "<|IMAGE|>",\n    "audio_token": "<|AUDIO|>",\n    "video_token": "<|VIDEO|>",\n    "vision_bos_token": "<|vision_bos|>",\n    "vision_eos_token": "<|vision_eos|>",\n    "audio_bos_token": "<|audio_bos|>",\n    "audio_eos_token": "<|audio_eos|>"\n  },\n  "model_max_length": 32768,\n  "pad_token": "<|endoftext|>",\n  "processor_class": "Qwen2_5OmniProcessor",\n  "split_special_tokens": false,\n  "tokenizer_class": "Qwen2Tokenizer",\n  "unk_token": null,\n  "image_token": "<|IMAGE|>",\n  "audio_token": "<|AUDIO|>",\n  "video_token": "<|VIDEO|>",\n  "vision_bos_token": "<|vision_bos|>",\n  "vision_eos_token": "<|vision_eos|>",\n  "audio_bos_token": "<|audio_bos|>",\n  "audio_eos_token": "<|audio_eos|>"\n}', '1_Pooling/config.json': '{"embedding_dimension": 4, "pooling_mode": "lasttoken", "include_prompt": true}', 'tokenizer.json': '{"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}, "added_tokens": []}'}


def make_source(source):
    for name, text in ASSETS.items():
        path = source / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(text, encoding='utf-8')
    vocab = {chr(i): i - 33 for i in range(33,127)}
    vocab.update({'Ġ': 94, 'Ċ': 95})
    specials = [{'content': token, 'id': 96+i, 'special': True, 'single_word': False,
                 'lstrip': False, 'rstrip': False, 'normalized': False}
                for i, token in enumerate(['<|im_start|>', '<|im_end|>', '<|audio_bos|>', '<|audio_eos|>', '<|AUDIO|>', '<|endoftext|>'])]
    bytelevel = dict(type='ByteLevel', add_prefix_space=False, trim_offsets=False, use_regex=False)
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    tokenizer = dict(model=dict(type='BPE', vocab=vocab, merges=[], dropout=None, unk_token=None,
                     continuing_subword_prefix='', end_of_word_suffix='', fuse_unk=False,
                     byte_fallback=False, ignore_merges=False), added_tokens=specials,
                     normalizer={'type':'NFC'}, decoder=bytelevel, post_processor=bytelevel,
                     pre_tokenizer=dict(type='Sequence', pretokenizers=[dict(type='Split', pattern={'Regex':pattern}, behavior='Isolated', invert=False), bytelevel]))
    (source / 'tokenizer.json').write_text(json.dumps(tokenizer), encoding='utf-8')
    (source / 'config.json').write_text(json.dumps(CONFIG), encoding='utf-8')
    path = source / 'additional_chat_templates/sentence_transformers.jinja'
    path.parent.mkdir(exist_ok=True)
    path.write_text(TEMPLATE, encoding='utf-8')
    shapes = {'model.embed_tokens.weight': [128,4], 'model.norm.weight': [4],
              'audio_tower.audio_bos_eos_token.weight': [2,4],
              'audio_tower.conv1.weight': [4,128,3], 'audio_tower.conv1.bias': [4],
              'audio_tower.conv2.weight': [4,4,3], 'audio_tower.conv2.bias': [4],
              'audio_tower.ln_post.weight': [4], 'audio_tower.ln_post.bias': [4],
              'audio_tower.proj.weight': [4,4], 'audio_tower.proj.bias': [4]}
    for key in ('input_layernorm', 'post_attention_layernorm'):
        shapes['model.layers.0.'+key+'.weight'] = [4]
    for key, shape in [('q', [4,4]), ('k',[2,4]), ('v',[2,4]), ('o',[4,4])]:
        base = 'model.layers.0.self_attn.'+key+'_proj'
        shapes[base+'.weight'] = shape
        if key != 'o': shapes[base+'.bias'] = [shape[0]]
    for key, shape in [('gate',[8,4]), ('up',[8,4]), ('down',[4,8])]:
        shapes['model.layers.0.mlp.'+key+'_proj.weight'] = shape
    for key in ('self_attn_layer_norm', 'final_layer_norm'):
        for kind in ('weight', 'bias'): shapes['audio_tower.layers.0.'+key+'.'+kind] = [4]
    for key in ('q','k','v','out'):
        base = 'audio_tower.layers.0.self_attn.'+key+'_proj'
        shapes[base+'.weight'] = [4,4]
        if key != 'k': shapes[base+'.bias'] = [4]
    for key, shape in [('fc1',[8,4]), ('fc2',[4,8])]:
        shapes['audio_tower.layers.0.'+key+'.weight'] = shape
        shapes['audio_tower.layers.0.'+key+'.bias'] = [shape[0]]
    # Known unused modalities are deliberately present and must be omitted.
    shapes['visual.patch_embed.proj.weight'] = [2,2]
    shapes['lm_head.weight'] = [8,4]
    write_checkpoint(source / 'model.safetensors', shapes)
    return shapes


def write_checkpoint(path, shapes):
    import math
    header, body = {}, bytearray()
    for name, shape in shapes.items():
        data = struct.pack('<H', 0x3f00) * math.prod(shape)
        header[name] = dict(dtype='BF16', shape=shape, data_offsets=[len(body),len(body)+len(data)])
        body.extend(data)
    encoded = json.dumps(header).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 8)
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+body)

if __name__ == '__main__':
    unittest.main()
