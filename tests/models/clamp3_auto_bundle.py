#!/usr/bin/env python3
"""Automatic pinned dependency resolution, exercised against a real offline HF cache."""
import json
from pathlib import Path
import subprocess
import sys
from mert_cache import install, REVISION
from q8_reference import tensors

build,out=map(Path,sys.argv[1:3]);root=Path(__file__).resolve().parents[2]
install(out/'auto-cache',out/'source')
path=out/'auto-bundle.gguf'
subprocess.run([sys.executable,str(root/'convert_hf_to_gguf.py'),str(out/'pipeline/model.pth'),str(path),
    '--clamp3-config',str(out/'pipeline/config.json'),'--tokenizer',str(build/'clamp3-fixture/tokenizer.json')],check=True)
stored={name for name,_,_ in tensors(path)}
assert any(name.startswith('mert.') for name in stored), 'CLaMP3 conversion did not automatically bundle MERT'
print('Automatic pinned MERT cache resolution passed')
