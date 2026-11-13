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

from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING
import os


def _load_backend():
    requested = os.environ.get("CLAP_CPP_BACKEND", "auto").lower()
    choices = ("cuda", "rocm", "vulkan")
    if requested not in ("auto", "cpu", *choices):
        raise ImportError("CLAP_CPP_BACKEND must be auto, cpu, cuda, rocm or vulkan")
    if requested == "auto":
        installed = [name for name in choices if find_spec(f"clap_cpp_{name}") is not None]
        if len(installed) > 1:
            raise ImportError(
                "Multiple clap-cpp GPU backends are installed; select one with "
                "CLAP_CPP_BACKEND=cpu|cuda|rocm|vulkan"
            )
        requested = installed[0] if installed else "cpu"
    if requested == "cpu":
        return requested, import_module(".clap", __name__)
    package = f"clap_cpp_{requested}"
    if find_spec(package) is None:
        raise ImportError(f'Install the backend with pip install "clap-cpp[{requested}]"')
    if version(f"clap-cpp-{requested}") != version("clap-cpp"):
        raise ImportError(f"clap-cpp and clap-cpp-{requested} must have the same version")
    try:
        return requested, import_module(f"{package}.clap")
    except (ImportError, OSError) as error:
        raise ImportError(
            f"Cannot load clap-cpp {requested} backend. Install its GPU runtime/driver "
            "or set CLAP_CPP_BACKEND=cpu."
        ) from error


backend, _native = _load_backend()
if TYPE_CHECKING:
    from .clap import CLAP as CLAP
else:
    CLAP = _native.CLAP

__all__ = ["CLAP", "backend"]
