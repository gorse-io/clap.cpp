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

import argparse
from pathlib import Path

from conversion import get_model_architecture, get_model_class


def convert(source, destination, dtype='f32', clamp3_config=None, tokenizer=None, mert2_revision=None, muq_revision=None, mert_revision=None, muq_mulan_revision=None, muq_config=None, text_config=None, clap2_revision=None):
    source = Path(source)
    model_architecture = get_model_architecture(source, clamp3_config, muq_revision, muq_mulan_revision)
    model_class = get_model_class(model_architecture)
    model_instance = model_class(source, destination, dtype, clamp3_config=clamp3_config,
                                 tokenizer=tokenizer, mert2_revision=mert2_revision, muq_revision=muq_revision,
                                 mert_revision=mert_revision, muq_mulan_revision=muq_mulan_revision,
                                 muq_config=muq_config, text_config=text_config, clap2_revision=clap2_revision)
    model_instance.write()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='CLAP/CLAP2/MERT/MuQ/MuLan directory or CLaMP3 .pth checkpoint')
    parser.add_argument('destination', type=Path)
    parser.add_argument('--dtype', choices=['f32', 'f16'], default='f32')
    parser.add_argument('--clamp3-config', type=Path,
                        help='CLaMP3 combined config JSON (otherwise infer from sibling XLM-R config)')
    parser.add_argument('--tokenizer', type=Path, help='CLAP2/CLaMP3/MuLan local tokenizer.json (default: beside checkpoint)')
    parser.add_argument('--clap2-revision', help='full HF revision SHA for CLAP2 metadata (default: pinned revision for the detected model size)')
    parser.add_argument('--mert2-revision', help='full pinned Hugging Face revision SHA for MERT2 metadata')
    parser.add_argument('--muq-revision', help='full pinned Hugging Face revision SHA for MuQ metadata')
    parser.add_argument('--mert-revision', help='full pinned Hugging Face revision SHA for MERT-v1 metadata')
    parser.add_argument('--muq-mulan-revision', help='full pinned MuQ-MuLan Hugging Face revision SHA')
    parser.add_argument('--muq-config', type=Path, help='local MuQ dependency config.json')
    parser.add_argument('--text-config', type=Path, help='local XLM-R dependency config.json')
    return parser.parse_args()


def main():
    args = parse_args()
    convert(args.source, args.destination, args.dtype, args.clamp3_config, args.tokenizer, args.mert2_revision, args.muq_revision, args.mert_revision, args.muq_mulan_revision, args.muq_config, args.text_config, args.clap2_revision)


if __name__ == '__main__':
    main()
