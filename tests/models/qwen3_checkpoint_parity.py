"""Optional published-weight parity against upstream Transformers Qwen3 modules.

Usage: python qwen3_checkpoint_parity.py BUILD LOCAL_HF_DIRECTORY OUTPUT.json
No downloads. The reference worker loads one decoder layer at a time in F32,
then exits before native inference starts, keeping heavyweight jobs serialized.
"""

import json
import math
from pathlib import Path
import subprocess
import sys

TEXTS = [
    'The capital of China is Beijing.',
    '中国的首都是北京。',
    'café cafe\u0301 🙂',
    '<|im_start|>user\nhello<|im_end|>',
]


def reference(source, output):
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    )

    torch.set_num_threads(2)
    config = Qwen3Config.from_json_file(str(source / 'config.json'))
    config._attn_implementation = 'eager'
    tokenizer = Tokenizer.from_file(str(source / 'tokenizer.json'))
    hf_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    results = []
    with torch.inference_mode(), safe_open(
        source / 'model.safetensors', framework='pt', device='cpu'
    ) as weights:
        prefix = '' if 'embed_tokens.weight' in weights.keys() else 'model.'
        states = []
        table = weights.get_slice(prefix + 'embed_tokens.weight')
        rope = Qwen3RotaryEmbedding(config=config)
        for text in TEXTS:
            ids = tokenizer.encode(text).ids
            assert ids == hf_tokenizer(text)['input_ids']
            # Gather only used rows; keep the four short sequences in memory.
            x = torch.stack(
                [table[token : token + 1].squeeze(0).float() for token in ids]
            ).unsqueeze(0)
            pos = torch.arange(len(ids)).unsqueeze(0)
            mask = torch.full((len(ids), len(ids)), float('-inf')).triu(1)[None, None]
            states.append(
                dict(
                    text=text, tokens=ids, x=x, pos=pos, mask=mask, position_embeddings=rope(x, pos)
                )
            )
        # Read each large layer exactly once for all texts, then release it.
        for i in range(config.num_hidden_layers):
            layer = Qwen3DecoderLayer(config, i).float().eval()
            p = prefix + f'layers.{i}.'
            layer.load_state_dict(
                {name: weights.get_tensor(p + name).float() for name in layer.state_dict()},
                strict=True,
            )
            for state in states:
                state['x'] = layer(
                    state['x'],
                    attention_mask=state['mask'],
                    position_ids=state['pos'],
                    position_embeddings=state['position_embeddings'],
                    use_cache=False,
                )[0]
            del layer
            if (i + 1) % 7 == 0:
                print('Transformers reference layers completed:', i + 1, flush=True)
        norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight.copy_(weights.get_tensor(prefix + 'norm.weight').float())
        for state in states:
            embedding = (
                torch.nn.functional.normalize(norm(state['x'])[:, -1], dim=-1).squeeze(0).tolist()
            )
            results.append(dict(text=state['text'], tokens=state['tokens'], embedding=embedding))
    output.write_text(
        json.dumps(
            dict(
                transformers=__import__('transformers').__version__,
                torch=torch.__version__,
                cases=results,
            ),
            ensure_ascii=False,
        )
    )


def main():
    build, source, output = map(Path, sys.argv[1:4])
    if len(sys.argv) == 5 and sys.argv[4] == '--reference':
        reference(source, output)
        return
    reference_path = output.with_suffix('.reference.json')
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(build),
            str(source),
            str(reference_path),
            '--reference',
        ],
        check=True,
    )
    refs = json.loads(reference_path.read_text())
    results = []
    for case in refs['cases']:
        run = subprocess.run(
            [
                str(build / 'clap'),
                'embedding',
                '--model',
                str(source / 'model-f16.gguf'),
                '--text',
                case['text'],
            ],
            text=True,
            capture_output=True,
        )
        if run.returncode:
            raise RuntimeError(run.stderr)
        actual = json.loads(run.stdout)
        ref = case['embedding']
        assert len(actual) == len(ref) == 1024
        error = max(abs(a - b) for a, b in zip(actual, ref))
        norm = math.sqrt(sum(v * v for v in actual))
        ref_norm = math.sqrt(sum(v * v for v in ref))
        cosine = sum(a * b for a, b in zip(actual, ref)) / (norm * ref_norm)
        record = dict(
            text=case['text'],
            tokens=case['tokens'],
            dimension=len(actual),
            max_abs_error=error,
            cosine_similarity=cosine,
            norm=norm,
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        results.append(record)
        assert error < 0.003 and cosine > 0.9999 and abs(norm - 1) < 2e-5, record
    output.write_text(
        json.dumps(
            dict(transformers=refs['transformers'], torch=refs['torch'], cases=results),
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
