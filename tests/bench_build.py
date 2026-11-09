"""Verify that only the CLI's license, not Benchmark SDKs, is installed."""
from pathlib import Path
import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory(prefix="clap-bench-install-") as directory:
    prefix = Path(directory)
    command = [sys.argv[1], "--install", sys.argv[2], "--prefix", str(prefix)]
    if len(sys.argv) > 3 and sys.argv[3]:
        command += ["--config", sys.argv[3]]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    # CMAKE_INSTALL_BINDIR is configurable (the libraries preset uses clap_cpp).
    assert any(path.is_file() and path.name in {"clap", "clap.exe"}
               for path in prefix.rglob("*")), "CLI not installed"
    assert len(list(prefix.rglob("benchmark.txt"))) == 1, "Benchmark license missing"
    unexpected = [str(path.relative_to(prefix)) for path in prefix.rglob("*")
                  if "benchmark" in path.name.lower() and path.name != "benchmark.txt"]
    assert not unexpected, f"Benchmark leaked into installed SDK: {unexpected}"
print("CLI and Benchmark license installed; no Benchmark archives, headers or CMake packages")
