"""Same-handle native overlap and public API surface regression."""
from pathlib import Path
import subprocess
import sys

from tiny_model import write_model

executable, destination, header = sys.argv[1:]
text = Path(header).read_text(encoding='utf-8')
for obsolete in ('clap_model_load', 'clap_model_free', 'clap_context_create',
                 'clap_default_context_options', 'clap_context_options'):
    assert obsolete not in text, f'public execution API leaked: {obsolete}'
root = Path(destination)
root.mkdir(parents=True, exist_ok=True)
model = root / 'model.gguf'
write_model(model)
subprocess.run([executable, str(model)], check=True)
