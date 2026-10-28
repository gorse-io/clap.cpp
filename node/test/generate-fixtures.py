"""Generate deterministic real models/audio using the shared stdlib fixture."""
import importlib.util
import json
from pathlib import Path
import struct
import wave

root = Path(__file__).resolve().parents[1]
source = root.parent / 'python' / 'tests' / 'tiny_models.py'
if not source.exists():
    source = root / 'native-tests' / 'tiny_models.py'
spec = importlib.util.spec_from_file_location('tiny_models', source)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
out = root / 'test' / 'fixtures'
out.mkdir(parents=True, exist_ok=True)
fixtures.clap_fixture(out / 'clap.gguf')
fixtures.mert_fixture(out / 'mert.gguf')
with wave.open(str(out / 'stereo.wav'), 'wb') as wav:
    wav.setparams((2, 2, 24000, 0, 'NONE', 'not compressed'))
    wav.writeframes(struct.pack('<hh', 16384, 0) * 24000)
# NumPy is optional: numerical oracle when available, never needed for fixtures.
try:
    ids = [0, 109, 106, 113, 113, 116, 2]
    values = fixtures.text_reference(ids).tolist()
except ImportError:
    (out / 'reference.json').unlink(missing_ok=True)
    print('NumPy unavailable: skipping optional independent text oracle')
else:
    (out / 'reference.json').write_text(json.dumps(dict(ids=ids, values=values)))
print(f'Generated CLAP/MERT/WAV fixtures in {out}')
