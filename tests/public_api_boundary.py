"""Keep upper inference consumers on the installed C interface.

Implementation/graph tests deliberately retain private-header access. Follow local
headers too so moving a private include to a convenience wrapper cannot bypass
this boundary; quantization implementation is not an inference consumer.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = (
    "src/clap.cpp", "src/embedding.cpp", "src/server.cpp",
    "node/bindings.cpp", "python/bindings.cpp", "rust/native/bridge.cpp",
)
PRIVATE_HEADERS = {
    "clap-internal.h", "model.h", "model-loader.h", "graph.h", "tensor-data.h", "vocab.h",
}
INCLUDES = re.compile(r'^\s*#\s*include\s*[<"]([^>\"]+)[>\"]', re.MULTILINE)


def inspect(path, seen):
    path = path.resolve()
    if path in seen:
        return False
    seen.add(path)
    source = path.read_text()
    assert not re.search(r'\bclap\s*::\s*CLAP\b', source), f"legacy C++ facade: {path}"
    public = False
    for name in INCLUDES.findall(source):
        header = Path(name)
        assert header.name not in PRIVATE_HEADERS and "models" not in header.parts, (
            f"private inference header {name} included by {path.relative_to(ROOT)}"
        )
        if header.name == "clap.h":
            public = True
            continue
        for directory in (path.parent, ROOT / "include", ROOT / "src"):
            target = (directory / header).resolve()
            if target.is_file() and target.is_relative_to(ROOT):
                public = inspect(target, seen) or public
                break
    return public


if __name__ == "__main__":
    for consumer in CONSUMERS:
        assert inspect(ROOT / consumer, set()), f"{consumer} does not consume clap.h"
    # No upper test can accidentally resurrect the removed C++ facade either.
    for path in (ROOT / "tests").rglob("*"):
        if path.suffix in {".h", ".c", ".cpp"}:
            source = path.read_text()
            assert not re.search(r'\bclap\s*::\s*CLAP\b', source), str(path)
            assert "clap-internal.h" not in INCLUDES.findall(source), str(path)
    print("public C API consumer boundary passed")
