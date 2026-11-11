"""Optional standard-GGUF parity against a separately built, pinned llama.cpp.

Usage: python qwen3_checkpoint_parity.py BUILD MODEL.gguf LLAMA_REFERENCE OUTPUT.json
Build qwen3-llama-reference.cpp against external llama.cpp as documented in
QWEN3_SOURCE.md. No downloads, Python ML packages, conversion or quantization.
The independent reference process exits before native model execution begins.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

TEXTS = [
    '',
    'The capital of China is Beijing.',
    '中国的首都是北京。',
    'café cafe\u0301 🙂',
    '<|im_start|>user\nhello<|im_end|>',
    '<|endoftext|>',
    '<|im_start|><|im_end|>',
    "I'RE_a\n\n12345",
    ' العربية हिन्दी 日本語 한글\t\r\n',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['build', 'model', 'reference', 'output']:
        parser.add_argument(name, type=lambda x: Path(x).resolve())
    parser.add_argument('--dimension', type=int, default=1024)
    parser.add_argument('--activation-precision', choices=['f32', 'model-default'], default='f32')
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--texts', type=Path, help='Optional JSON array for bounded probes')
    parser.add_argument('--reference-f32-operands', action='store_true')
    parser.add_argument('--api-only', action='store_true', help='Do not compare default CLI to an F32-operand oracle')
    parser.add_argument('--sha256', help='Expected immutable published GGUF digest')
    parser.add_argument('--revision', help='Published GGUF repository revision')
    args = parser.parse_args()
    if args.reference_f32_operands and (args.activation_precision != 'f32' or not args.api_only):
        parser.error('--reference-f32-operands requires --activation-precision f32 --api-only')
    if args.dimension <= 0 or args.context <= 0:
        parser.error('dimension and context must be positive')
    build, model, reference, output = args.build, args.model, args.reference, args.output
    texts = json.loads(args.texts.read_text(encoding='utf-8')) if args.texts else TEXTS
    hasher = hashlib.sha256()
    with model.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if args.sha256 and digest != args.sha256:
        raise ValueError('published GGUF SHA-256 mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus = output.with_suffix('.texts.json')
    corpus.write_text(json.dumps(texts, ensure_ascii=False), encoding='utf-8')
    command = [str(reference), str(model), str(corpus)]
    if args.reference_f32_operands:
        command.append('--f32-operands')
    run = subprocess.run(command, capture_output=True, text=True)
    output.with_suffix('.llama.log').write_text(run.stderr, encoding='utf-8')
    if run.returncode:
        raise RuntimeError(run.stderr)
    cases = json.loads(run.stdout)
    assert len(cases) == len(texts) and [c['text'] for c in cases] == texts
    fixture = output.with_suffix('.fixture.json')
    fixture.write_text(json.dumps(dict(sample_rate=0, projection_dim=args.dimension, context_length=args.context,
                                      tolerance=0.003, model_default_activations=args.activation_precision == 'model-default',
                                      text_cases=cases, token_cases=[])), encoding='utf-8')
    # Exact llama.cpp token IDs plus real string/token C API embeddings and modality errors.
    subprocess.run([str(build / 'test-qwen3-model'), str(model), str(fixture), str(output.with_suffix('.native.json'))], check=True)
    native_cases = json.loads(output.with_suffix('.native.json').read_text(encoding='utf-8'))
    results = []
    for case in ([] if args.api_only else cases):
        run = subprocess.run([str(build / 'clap'), 'embedding', '--model', str(model), '--text', case['text']],
                             capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        actual = json.loads(run.stdout)
        ref = case['expected']
        assert len(actual) == len(ref) == args.dimension
        error = max(abs(a - b) for a, b in zip(actual, ref))
        norm = math.sqrt(sum(v * v for v in actual))
        ref_norm = math.sqrt(sum(v * v for v in ref))
        cosine = sum(a * b for a, b in zip(actual, ref)) / (norm * ref_norm)
        record = dict(text=case['text'], tokens=case['tokens'], dimension=len(actual), max_abs_error=error,
                      cosine_similarity=cosine, norm=norm, relative_l2=math.sqrt(sum((a-b)**2 for a, b in zip(actual, ref))) / ref_norm,
                      actual=actual, expected=ref)
        print(json.dumps({k: v for k, v in record.items() if k not in ['actual', 'expected']}, ensure_ascii=False), flush=True)
        results.append(record)
        assert error < 0.003 and cosine > 0.9999 and record['relative_l2'] < 0.02 and abs(norm - 1) < 2e-5, record
    output.write_text(json.dumps(dict(model=str(model), bytes=model.stat().st_size, sha256=digest,
                                      reference=str(reference), revision=args.revision, dimension=args.dimension,
                                      activation_precision=args.activation_precision, reference_f32_operands=args.reference_f32_operands,
                                      cli_checked=not args.api_only, native_cases=native_cases, cases=results), ensure_ascii=False, indent=2) + '\n',
                      encoding='utf-8')


if __name__ == '__main__':
    main()
