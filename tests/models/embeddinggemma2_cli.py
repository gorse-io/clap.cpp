"""Text-only EmbeddingGemma2 CLI/HTTP integration, stdlib fixtures."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from embeddinggemma2 import fixture, save
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embeddings import running_server, request


def check(actual, expected):
    assert len(actual) == len(expected)
    assert all(math.isfinite(v) for v in actual)
    assert abs(sum(v*v for v in actual) - 1) < 1e-5
    assert max(abs(a-b) for a, b in zip(actual, expected)) < 5e-6


def main(binary):
    help_output = subprocess.run([binary, "embedding", "--help"], check=True,
                                 capture_output=True, text=True).stdout
    assert "--mmproj" in help_output, "audio projector option is missing"
    with tempfile.TemporaryDirectory(prefix="embeddinggemma2-cli-") as directory:
        for storage in ("f32", "f16"):
            metadata, tensors, oracle = fixture(storage)
            model = Path(directory) / (storage + ".gguf")
            save(model, metadata, tensors, storage)
            cases = [case for case in oracle["text_cases"] if case["text"]]
            for case in cases:
                result = subprocess.run([binary, "embedding", "--model", str(model),
                                         "--threads", "2", "--text", case["text"]],
                                        check=True, capture_output=True, text=True, timeout=30)
                check(json.loads(result.stdout), case["embedding"])
            with running_server(binary, model) as url:
                status, result = request(url, {"input": [case["text"] for case in cases]})
                assert status == 200, result
                assert len(result["data"]) == len(cases)
                for i, (entry, case) in enumerate(zip(result["data"], cases)):
                    assert entry["index"] == i
                    check(entry["embedding"], case["embedding"])
    print("EmbeddingGemma2 F32/F16 CLI and HTTP match independent scalar oracle")


if __name__ == "__main__":
    main(sys.argv[1])
