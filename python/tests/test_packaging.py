# Copyright 2026 clap.cpp Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
try:
    import tomllib
except ImportError:  # Python 3.9/3.10
    import tomli as tomllib
import zipfile

from jinja2 import Template
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
CUDA_DEPENDENCIES = [
    'nvidia-cuda-runtime-cu12==12.9.79; sys_platform == "linux"',
    'nvidia-cublas-cu12==12.9.2.10; sys_platform == "linux"',
    'nvidia-cuda-nvrtc-cu12==12.9.86; sys_platform == "linux"',
]


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm", "vulkan"])
def test_backend_metadata_dependencies_are_cuda_only(backend):
    rendered = Template((ROOT / "python/pyproject.toml.j2").read_text()).render(backend=backend)
    metadata = tomllib.loads(rendered)
    assert metadata["project"]["dependencies"] == (CUDA_DEPENDENCIES if backend == "cuda" else [])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux CUDA workflow uses POSIX shell tools")
def test_python_cuda_build_and_repair_use_external_nvidia_dependencies(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    steps = workflow["jobs"]["python-build-cuda"]["steps"]
    commands = {step["name"]: step["run"] for step in steps if "run" in step}
    # Execute the actual shell blocks with tool-boundary recorders: assert argv,
    # not just incidental strings in the workflow.
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "commands"
    for name in ("cmake", "auditwheel"):
        tool = tools / name
        tool.write_text('#!/bin/sh\nprintf "%s\\n" "$0 $*" >> "$COMMAND_LOG"\n')
        tool.chmod(0o755)
    env = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", COMMAND_LOG=str(log),
               CMAKE_ARGS="", CMAKE_BUILD_PARALLEL_LEVEL="1")
    for step in ("Compile external CUDA shared runtime", "Repair CUDA wheel"):
        subprocess.run(["bash", "-c", commands[step]], cwd=tmp_path, env=env, check=True)
    lines = log.read_text().splitlines()
    assert "-DGGML_STATIC=OFF" in lines[0]
    assert "-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared" in lines[0]
    for library in ("libcuda.so.1", "libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
        assert f"--exclude {library}" in lines[-1]


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm", "vulkan"])
def test_backend_distribution_inventory_and_tags(tmp_path, backend):
    project = tmp_path / "project"
    project.mkdir()
    package = "clap_cpp" + (f"_{backend}" if backend != "cpu" else "")
    shutil.copytree(ROOT / "python/clap_cpp", project / package)
    for name in ("README.md", "LICENSE"):
        shutil.copyfile(ROOT / "python" / name, project / name)
    (project / "pyproject.toml").write_text(
        Template((ROOT / "python/pyproject.toml.j2").read_text()).render(backend=backend))
    # Deliberately stage fake native bytes and forbidden inputs; this exercises
    # archive assembly, NOT native loading, ELF validity or auditwheel repair.
    (project / package / "libclap.so").write_bytes(b"\x7fELFfixture")
    (project / package / "licenses").mkdir()
    (project / package / "licenses/clap-cpp.txt").write_text("fixture notice")
    (project / package / "forbidden.j2").write_text("template")
    (project / package / "tests").mkdir()
    (project / package / "tests/test_leak.py").write_text("raise AssertionError")
    (project / "pyproject.toml.j2").write_text("template")
    subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--sdist", "--wheel", str(project)],
                   check=True, capture_output=True, text=True)
    sdist, = (project / "dist").glob("*.tar.gz")
    wheel, = (project / "dist").glob("*.whl")
    with tarfile.open(sdist) as archive:
        source_names = archive.getnames()
        assert any(name.endswith("/pyproject.toml") for name in source_names)
        assert not any(name.endswith(".so") or "/licenses/" in name for name in source_names)
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        assert f"{package}/libclap.so" in wheel_names
        assert f"{package}/py.typed" in wheel_names
        assert f"{package}/clap.pyi" in wheel_names
        info = next(name for name in wheel_names if name.endswith("/WHEEL"))
        tags = archive.read(info).decode()
        assert "Root-Is-Purelib: false" in tags
        assert "Tag: py3-none-" in tags and "Tag: py3-none-any" not in tags
        info = next(name for name in wheel_names if name.endswith("/METADATA"))
        metadata = archive.read(info).decode()
        for requirement in CUDA_DEPENDENCIES:
            assert (f"Requires-Dist: {requirement}" in metadata) == (backend == "cuda")
    for name in source_names + wheel_names:
        assert not name.endswith((".j2", ".pyc"))
        assert "tests" not in Path(name).parts
