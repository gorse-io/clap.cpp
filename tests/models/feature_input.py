"""Test-only access to private native feature encoders, not a public CLI path."""
import json
from pathlib import Path
import subprocess

def encode(build, model, features, frames, output):
    path = Path(output)/'private-features.json'
    path.write_text(json.dumps({'values': list(map(float, features)), 'frames': int(frames)}))
    return json.loads(subprocess.check_output([str(Path(build)/'feature-input-test'),str(model),str(path)]))
