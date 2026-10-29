"""Verify the installed distribution's public typing contract."""

from importlib.resources import files
from pathlib import Path
import subprocess
import sys


def test_installed_package_has_typing_metadata():
    package = files("clap_cpp")
    assert package.joinpath("py.typed").is_file()
    assert package.joinpath("_clap_cpp.pyi").is_file()


def test_public_api_types():
    result = subprocess.run(
        [
            sys.executable, "-m", "mypy", "--strict", "--no-incremental",
            str(Path(__file__).with_name("typing_api.py")),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
