#!/usr/bin/env python3
"""Breaking public interface contract: waveform input, no feature/source flags."""
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parents[2]
cli = Path(sys.argv[1])
help_text = subprocess.check_output([str(cli), 'embedding', '--help'], text=True)
assert '--audio-features' not in help_text, 'obsolete public feature CLI remains'
assert '--mert-model' not in help_text, 'obsolete external-model fallback remains'
help_text = subprocess.check_output([sys.executable, str(root/'convert_hf_to_gguf.py'), '--help'], text=True)
assert '--mert-source' not in help_text, 'bundling must be automatic, not optional'
header = (root/'include/clap.h').read_text()
assert 'audio_features(' not in header, 'obsolete public feature API remains'
print('Waveform-only public API/CLI and automatic bundle surface passed')
