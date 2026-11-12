"""Independent soft-token transformer regression, with generated weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from embeddinggemma2 import fixture, save

with tempfile.TemporaryDirectory(prefix="gemma-soft-") as directory:
    for storage in ("f32", "f16"):
        metadata, tensors, oracle = fixture(storage)
        model = Path(directory) / (storage + ".gguf")
        reference = model.with_suffix(".json")
        save(model, metadata, tensors, storage)
        reference.write_text(json.dumps(oracle), encoding="utf-8")
        subprocess.run([sys.argv[1], str(model), str(reference)], check=True)
