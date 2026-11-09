"""Test-only access to private native feature encoders, not a public CLI path."""
import json
from pathlib import Path
import subprocess

def extract(build, model, audio):
    """Decode/resample audio and run private MERT all-layer CLaMP3 pooling.

    test-feature-extract MERT.gguf AUDIO emits {values, frames, dimension} JSON;
    two inference threads are requested by FeatureTestAccess, with normal runtime device selection.
    Build with BUILD_TESTING=ON and MERT_MODEL_PARITY=ON. Never calls clap embedding.
    """
    return json.loads(subprocess.check_output([
        str(Path(build)/'test-feature-extract'), str(model), str(audio)]))


def encode(build, model, features, frames, output):
    path = Path(output)/'private-features.json'
    path.write_text(json.dumps({'values': list(map(float, features)), 'frames': int(frames)}))
    return json.loads(subprocess.check_output([str(Path(build)/'test-feature-input'),str(model),str(path)]))
