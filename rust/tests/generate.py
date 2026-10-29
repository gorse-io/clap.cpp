"""Generate real tiny models using the repository's stdlib-only fixtures."""
import importlib.util
from pathlib import Path
import struct
import math


def text_reference(ids):
    """Scalar stdlib oracle; mirrors the existing fixture's independent forward pass.

    No Rust/C++ calls or ML libraries. Weights round through on-disk F32 first.
    """
    def weight(name, n):
        norm = 'norm' in name.lower() or 'LayerNorm' in name
        values = [(1.0 if name.endswith('weight') else 0.0) if norm else
                  0.12 * math.sin(i * 0.73 + sum(name.encode()) * 0.17)
                  for i in range(n)]
        return list(struct.unpack('<' + 'f' * n, struct.pack('<' + 'f' * n, *values)))

    def linear(x, prefix, output):
        w = weight(prefix + '.weight', len(x) * output)
        b = weight(prefix + '.bias', output)
        return [sum(a * v for a, v in zip(x, w[i*len(x):(i+1)*len(x)])) + b[i]
                for i in range(output)]

    def norm(x, prefix):
        mean = sum(x) / len(x)
        var = sum((a-mean)**2 for a in x) / len(x)
        w, b = weight(prefix + '.weight', len(x)), weight(prefix + '.bias', len(x))
        return [(a-mean) / math.sqrt(var+1e-5) * s + t for a, s, t in zip(x, w, b)]

    def add(x, y):
        return [a+b for a,b in zip(x,y)]

    word = weight('text.embeddings.word_embeddings.weight', 261*4)
    pos = weight('text.embeddings.position_embeddings.weight', 514*4)
    types = weight('text.embeddings.token_type_embeddings.weight', 4)
    x = [norm(add(add(word[t*4:(t+1)*4], pos[(i+2)*4:(i+3)*4]), types),
              'text.embeddings.LayerNorm') for i,t in enumerate(ids)]
    prefix = 'text.encoder.layer.0'
    q,k,v = [[linear(row, prefix + '.attention.self.' + part, 4) for row in x]
             for part in ['query', 'key', 'value']]
    attention = []
    for row in q:
        scores = [sum(a*b for a,b in zip(row, key))/2 for key in k]
        probs = [math.exp(a-max(scores)) for a in scores]
        total = sum(probs)
        attention.append([sum(p/total * val[i] for p,val in zip(probs,v)) for i in range(4)])
    x = [norm(add(row, linear(a, prefix + '.attention.output.dense', 4)),
              prefix + '.attention.output.LayerNorm') for row,a in zip(x,attention)]
    hidden = [linear(row, prefix + '.intermediate.dense', 8) for row in x]
    hidden = [[0.5*a*(1+math.erf(a/math.sqrt(2))) for a in row] for row in hidden]
    x = [norm(add(row, linear(h, prefix + '.output.dense', 4)), prefix + '.output.LayerNorm')
         for row,h in zip(x,hidden)]
    pooled = [math.tanh(a) for a in linear(x[0], 'text.pooler.dense', 4)]
    projected = linear([max(a,0) for a in linear(pooled, 'text_proj.linear1', 3)], 'text_proj.linear2', 3)
    length = math.sqrt(sum(a*a for a in projected))
    return [a/length for a in projected]
import sys
import wave

crate = Path(__file__).resolve().parents[1]
source = crate / 'vendor/test-data/tiny_models.py'
if not source.is_file():
    source = crate.parent / 'python/tests/tiny_models.py'
# Do not add bytecode artifacts to the vendored source archive during tests.
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('tiny_models', source)
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)
out = Path(sys.argv[1])
(out / 'text-reference.txt').write_text(' '.join(map(str, text_reference([0, 77, 110, 2]))))
tiny.clap_fixture(out / '模型.gguf')
tiny.mert_fixture(out / 'mert.gguf')
with wave.open(str(out / '音频.wav'), 'wb') as f:
    f.setparams((1, 2, 24000, 4, 'NONE', 'not compressed'))
    f.writeframes(struct.pack('<4h', 0, 8192, -16384, 24576))
with wave.open(str(out / '立体声.wav'), 'wb') as f:
    f.setparams((2, 2, 24000, 4, 'NONE', 'not compressed'))
    f.writeframes(struct.pack('<8h', -4096, 4096, 4096, 12288, -20480, -12288, 20480, 28672))

def wav(name, rate, samples):
    with wave.open(str(out / name), 'wb') as f:
        f.setparams((1, 2, rate, len(samples), 'NONE', 'not compressed'))
        f.writeframes(struct.pack('<' + 'h' * len(samples), *samples))

samples = [int(8192 * math.sin(i * 0.03)) for i in range(12 * 48000)]
wav('long.wav', 48000, samples)
wav('crop.wav', 48000, samples[48000:11 * 48000])
wav('mert.wav', 24000, [int(6553 * math.sin(i * 0.03)) for i in range(24000)])
(out / 'invalid.wav').write_bytes(b'not an audio file')
