#!/usr/bin/env python3
"""Convert local Hugging Face CLAP, CLaMP3 or MERT2 weights to native GGUF.

Conversion is the only step requiring Python and torch. Checkpoints are loaded
with weights_only=True and mmap=True. CLaMP3 symbolic encoder tensors are not
written because native symbolic input is not supported. MERT2 safetensors are
streamed one tensor at a time so the published 632M-parameter checkpoint can be
converted on memory-constrained hosts.
"""
import argparse
from pathlib import Path

from conversion import get_model_architecture, get_model_class


def convert(source, destination, dtype='f32', clamp3_config=None, tokenizer=None, mert2_revision=None):
    source = Path(source)
    model_architecture = get_model_architecture(source, clamp3_config)
    model_class = get_model_class(model_architecture)
    model_instance = model_class(source, destination, dtype, clamp3_config=clamp3_config,
                                 tokenizer=tokenizer, mert2_revision=mert2_revision)
    model_instance.write()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='CLAP directory or CLaMP3 .pth checkpoint')
    parser.add_argument('destination', type=Path)
    parser.add_argument('--dtype', choices=['f32', 'f16'], default='f32')
    parser.add_argument('--clamp3-config', type=Path,
                        help='CLaMP3 combined config JSON (otherwise infer from sibling XLM-R config)')
    parser.add_argument('--tokenizer', type=Path, help='CLaMP3 tokenizer.json (default: beside checkpoint)')
    parser.add_argument('--mert2-revision', help='full pinned Hugging Face revision SHA for MERT2 metadata')
    return parser.parse_args()


def main():
    args = parse_args()
    convert(args.source, args.destination, args.dtype, args.clamp3_config, args.tokenizer, args.mert2_revision)


if __name__ == '__main__':
    main()
