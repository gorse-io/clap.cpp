"""Optional standard-GGUF parity against a separately built, pinned llama.cpp.

Usage: python qwen3_checkpoint_parity.py BUILD MODEL.gguf LLAMA_REFERENCE OUTPUT.json
Build qwen3-llama-reference.cpp against external llama.cpp as documented in
QWEN3_SOURCE.md. No downloads, Python ML packages, conversion or quantization.
The independent reference process exits before native model execution begins.
"""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

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
    build, model, reference, output = map(lambda x: Path(x).resolve(), sys.argv[1:])
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus = output.with_suffix('.texts.json')
    corpus.write_text(json.dumps(TEXTS, ensure_ascii=False), encoding='utf-8')
    run = subprocess.run([str(reference), str(model), str(corpus)], capture_output=True, text=True)
    output.with_suffix('.llama.log').write_text(run.stderr, encoding='utf-8')
    if run.returncode:
        raise RuntimeError(run.stderr)
    cases = json.loads(run.stdout)
    assert len(cases) == len(TEXTS) and [c['text'] for c in cases] == TEXTS
    fixture = output.with_suffix('.fixture.json')
    fixture.write_text(json.dumps(dict(sample_rate=0, projection_dim=1024, context_length=32768,
                                      tolerance=0.003, text_cases=cases, token_cases=[])), encoding='utf-8')
    # Exact llama.cpp token IDs plus real string/token C API embeddings and modality errors.
    subprocess.run([str(build / 'test-qwen3-model'), str(model), str(fixture)], check=True)
    results = []
    for case in cases:
        run = subprocess.run([str(build / 'clap'), 'embedding', '--model', str(model), '--text', case['text']],
                             capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        actual = json.loads(run.stdout)
        ref = case['expected']
        assert len(actual) == len(ref) == 1024
        error = max(abs(a - b) for a, b in zip(actual, ref))
        norm = math.sqrt(sum(v * v for v in actual))
        ref_norm = math.sqrt(sum(v * v for v in ref))
        cosine = sum(a * b for a, b in zip(actual, ref)) / (norm * ref_norm)
        record = dict(text=case['text'], tokens=case['tokens'], dimension=len(actual), max_abs_error=error,
                      cosine_similarity=cosine, norm=norm)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        results.append(record)
        assert error < 0.003 and cosine > 0.9999 and abs(norm - 1) < 2e-5, record
    hasher = hashlib.sha256()
    with model.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    output.write_text(json.dumps(dict(model=str(model), bytes=model.stat().st_size, sha256=digest,
                                      reference=str(reference), cases=results), ensure_ascii=False, indent=2) + '\n',
                      encoding='utf-8')


if __name__ == '__main__':
    main()
