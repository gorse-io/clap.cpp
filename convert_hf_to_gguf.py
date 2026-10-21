#!/usr/bin/env python3
"""Convert local Hugging Face CLAP, CLaMP3 or MERT2 weights to native GGUF.

Conversion is the only step requiring Python and torch. Checkpoints are loaded
with weights_only=True and mmap=True. CLaMP3 symbolic encoder tensors are not
written because native symbolic input is not supported. MERT2 safetensors are
streamed one tensor at a time so the published 632M-parameter checkpoint can be
converted on memory-constrained hosts.
"""
import argparse
import json
from pathlib import Path

from conversion import clap, clamp3, mert2
from conversion.base import write_gguf


def convert(source, destination, dtype='f32', clamp3_config=None, tokenizer=None, mert2_revision=None):
    import numpy as np
    import torch
    source, destination = Path(source), Path(destination)
    raw_config = json.loads((source / 'config.json').read_text()) if source.is_dir() and (source / 'config.json').exists() else {}
    if raw_config.get('model_type') == 'mert2':
        checkpoint, metadata = mert2.mert2_inputs(source, mert2_revision)
        mert2.write_safetensors_gguf(destination, checkpoint, metadata, dtype, np, torch)
        return
    if clamp3_config is not None or source.is_file():
        tensors, metadata = clamp3.clamp3_inputs(source, Path(clamp3_config) if clamp3_config else None,
                                          Path(tokenizer) if tokenizer else None, torch)
    else:
        tensors, metadata = clap.clamp_inputs(source, torch)
    write_gguf(destination, tensors, metadata, dtype, np, torch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='CLAP directory or CLaMP3 .pth checkpoint')
    parser.add_argument('destination', type=Path)
    parser.add_argument('--dtype', choices=['f32', 'f16'], default='f32')
    parser.add_argument('--clamp3-config', type=Path,
                        help='CLaMP3 combined config JSON (otherwise infer from sibling XLM-R config)')
    parser.add_argument('--tokenizer', type=Path, help='CLaMP3 tokenizer.json (default: beside checkpoint)')
    parser.add_argument('--mert2-revision', help='full pinned Hugging Face revision SHA for MERT2 metadata')
    args = parser.parse_args()
    convert(args.source, args.destination, args.dtype, args.clamp3_config, args.tokenizer, args.mert2_revision)
