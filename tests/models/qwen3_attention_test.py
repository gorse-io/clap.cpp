"""Offline default-precision attention oracle, independent of native kernels."""
import json
from pathlib import Path
import subprocess
import sys
from qwen3_fixture import model_data, standard_metadata, write_gguf, oracle

build, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
# Short-query heads stay below the native F16 SIMD update width: ARM's
# half-vector FMA also rounds probabilities to half, unlike x86's F32 lanes.
# Use scalar updates here so this independent oracle is portable without
# accepting platform-specific alternative expected vectors.
for heads, kv, dim, lengths in [(4, 2, 16, [7, 31]), (4, 1, 128, [64, 65])]:
    c, t, weights = model_data(16, heads=heads, kv_heads=kv, head_dim=dim, depth=1)
    model = output / f'gqa-{heads}-{kv}-{dim}.gguf'
    write_gguf(model, standard_metadata(c, t), weights)
    for length in lengths:
        ids = [(i * 17 + 3) % 251 for i in range(length)]
        expected = oracle(c, weights, ids, attention_precision='default')
        unrounded = oracle(c, weights, ids)
        assert max(abs(a-b) for a,b in zip(expected, unrounded)) > 3e-5, 'K/V rounding sensitivity'
        if length < 64:
            rounded = oracle(c, weights, ids, attention_precision='rounded-query')
            assert max(abs(a-b) for a,b in zip(expected, rounded)) > 3e-5, 'F32 query sensitivity'
        fixture = dict(sample_rate=0, projection_dim=16, context_length=128, tolerance=3e-5,
                       model_default_activations=True, text_cases=[],
                       token_cases=[dict(name=f'gqa={heads}/{kv},dim={dim},tokens={length}',
                                         tokens=ids, expected=expected)])
        path = output / f'{heads}-{kv}-{dim}-{length}.json'
        for threads in [1, 2]:
            fixture['threads'] = threads
            path.write_text(json.dumps(fixture))
            subprocess.run([str(build / 'test-qwen3-model'), str(model), str(path)], check=True)
        fixture['model_default_activations'] = False
        fixture['token_cases'][0]['expected'] = unrounded
        path.write_text(json.dumps(fixture))
        subprocess.run([str(build / 'test-qwen3-model'), str(model), str(path)], check=True)
print('Default Qwen3 attention: short/long, grouped heads, rounded K/V and unrounded Q passed')
