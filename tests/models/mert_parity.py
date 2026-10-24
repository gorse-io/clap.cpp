#!/usr/bin/env python3
"""Offline numerical oracle using reviewed, pinned official MERT-v1 sources.
No network access or remote-code trust is performed by this test.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import wave
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

REVISION = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'

def reference(source):
    spec = importlib.util.spec_from_file_location('mert_reference', source / '__init__.py',
                                                submodule_search_locations=[str(source)])
    package = importlib.util.module_from_spec(spec)
    sys.modules['mert_reference'] = package
    package.__path__ = [str(source)]
    from mert_reference.configuration_MERT import MERTConfig
    from mert_reference.modeling_MERT import MERTModel
    return MERTConfig, MERTModel

def wav(path, pcm):
    # Float32 RIFF preserves the exact oracle waveform, including silence/DC.
    import struct
    data = np.asarray(pcm, dtype='<f4').tobytes()
    path.write_bytes(b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' +
                     struct.pack('<IHHIIHH', 16, 3, 1, 24000, 96000, 4, 32) +
                     b'data' + struct.pack('<I', len(data)) + data)

def error(actual, expected, label, limit):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    delta = actual - expected
    maximum = float(np.max(np.abs(delta)))
    print(label, json.dumps(dict(max_abs=maximum, rmse=float(np.sqrt(np.mean(delta**2))),
          cosine=float(np.dot(actual.ravel(), expected.ravel()) /
                       max(np.linalg.norm(actual)*np.linalg.norm(expected), 1e-30)))), flush=True)
    assert np.isfinite(actual).all() and maximum < limit, (label, maximum, limit)

def run(args):
    torch.set_num_threads(2); torch.manual_seed(71)
    args.out.mkdir(parents=True, exist_ok=True)
    Config, Network = reference(args.source)
    if args.checkpoint:
        config = Config.from_json_file(str(args.checkpoint/'config.json'))
        network = Network(config).eval()
        state = torch.load(args.checkpoint/'pytorch_model.bin', weights_only=True, mmap=True, map_location='cpu')
        network.load_state_dict(state, assign=True)
        source = args.checkpoint
    else:
        config = Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, conv_dim=[8,8,8,8,8,8,8], conv_kernel=[10,3,3,3,3,2,2],
            conv_stride=[5,2,2,2,2,2,2], num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2, hidden_dropout=0., attention_dropout=0.,
            feat_proj_dropout=0., layerdrop=0., mask_time_prob=0., mask_feature_prob=0.,
            feat_extract_norm='group', feat_proj_layer_norm=True, sample_rate=24000,
            feature_extractor_cqt=False, deepnorm=False, do_stable_layer_norm=False)
        network = Network(config).eval()
        source = args.out/'source'; source.mkdir(exist_ok=True)
        config.to_json_file(source/'config.json')
        Wav2Vec2FeatureExtractor(sampling_rate=24000, return_attention_mask=True,
                                do_normalize=True).save_pretrained(source)
        torch.save(network.state_dict(), source/'pytorch_model.bin')
    processor = Wav2Vec2FeatureExtractor.from_pretrained(source)
    root = Path(__file__).resolve().parents[2]
    rng = np.random.default_rng(719)
    cases = {'broadband': rng.normal(0.1,0.12,24000).astype('float32'),
             'tone': (0.2*np.sin(np.arange(24800)*0.071)+0.2).astype('float32'),
             'silence': np.zeros(24000, dtype='float32')}
    if not args.checkpoint:
        cases.update({'two_chunks':rng.normal(0.1,0.13,6*24000).astype('float32'),
                      'discard_tail':rng.normal(0.1,0.13,5*24000+23999).astype('float32')})
    for dtype in ['f32','f16']:
        target = args.out/f'model-{dtype}.gguf'
        subprocess.run([sys.executable, str(root/'convert_hf_to_gguf.py'),str(source),str(target),
                        '--dtype', dtype,'--mert-revision', REVISION], check=True)
        subprocess.run([str(args.build/'mert-model-test'), str(target), str(args.out)], check=True)
        cases['resampled'] = np.frombuffer((args.out/'resampled-control.wav').read_bytes()[44:], dtype='<f4').copy()
        # Compare identical storage-rounded weights, not a half-activation oracle.
        if dtype == 'f16':
            # Converter materializes weight norm before rounding; remove it here too.
            torch.nn.utils.parametrize.remove_parametrizations(network.encoder.pos_conv_embed.conv,
                                                               'weight',leave_parametrized=True)
            for p in network.parameters():
                if p.ndim > 1: p.data = p.data.half().float()
        for name, pcm in cases.items():
            audio = args.out/f'{name}.wav'; wav(audio,pcm)
            normalized = processor(pcm, sampling_rate=24000, return_tensors='pt').input_values
            with torch.no_grad():
                result = network(normalized, output_hidden_states=True)
            actual = json.loads(subprocess.check_output([str(args.build/'clap'), 'embedding','-m',str(target),
                                 '-a',str(audio),'--frames','-t','2'], text=True))
            error(np.asarray(actual['values']).reshape(actual['frames'],actual['dimension']),
                  result.last_hidden_state[0].numpy(), f'{dtype}/{name}/frames', 0.0002)
            if len(pcm) >= 24000:
                chunks=[]
                for start in range(0,len(pcm),120000):
                    chunk=normalized[:,start:start+120000]
                    if chunk.shape[-1] < 24000: break
                    with torch.no_grad():
                        states=network(chunk,output_hidden_states=True).hidden_states
                    chunks.append(torch.stack(states).mean(-2).mean(0)[0].numpy())
                actual = json.loads(subprocess.check_output([str(args.build/'clap'), 'embedding','-m',str(target),
                                     '-a',str(audio),'--clamp3-features','-t','2'], text=True))
                error(np.asarray(actual['values']).reshape(actual['frames'],actual['dimension']),
                      np.stack(chunks), f'{dtype}/{name}/clamp3', 0.00002)
                if name == 'resampled':
                    resampled = json.loads(subprocess.check_output([str(args.build/'clap'), 'embedding',
                        '-m',str(target),'-a',str(args.out/'resampled-input.wav'),
                        '--clamp3-features','-t','2'], text=True))
                    np.testing.assert_array_equal(resampled['values'], actual['values'])
                    print(f'{dtype}/48kHz CLI -> 24kHz MERT matches native-rate control', flush=True)
    for n in [0,399,23999]:
        path=args.out/f'short-{n}.wav';wav(path,np.zeros(n,dtype='float32'))
        result=subprocess.run([str(args.build/'clap'), 'embedding','-m',str(target),'-a',str(path),
                               '--clamp3-features'],capture_output=True,text=True)
        assert result.returncode != 0, (n,result.stdout)
    print('MERT-v1 numerical parity passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('build',type=Path)
    parser.add_argument('out',type=Path);parser.add_argument('source',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    run(parser.parse_args())
