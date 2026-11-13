"""Offline shared-model/context regression fixture (stdlib only)."""
import json
from pathlib import Path
import subprocess
import sys

from tiny_model import write_model

executable, destination, *flags = sys.argv[1:]
root = Path(destination)
root.mkdir(parents=True, exist_ok=True)
model = root / "model.gguf"
fixture = root / "fixture.json"
fixture.write_text(json.dumps(write_model(model)), encoding="utf-8")
result = subprocess.run([executable, str(model), str(fixture), *flags])
raise SystemExit(result.returncode)
