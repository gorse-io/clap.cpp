#!/usr/bin/env python3
"""Published MERT-v1-95M -> published CLaMP3 SAAS raw-waveform oracle.

All inputs are local, reviewed snapshots. No network or remote-code loading.
CLaMP3 GGUF must be F32; --mert-dtype selects MERT storage rounding.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from transformers import BertConfig, BertModel, Wav2Vec2FeatureExtractor
from mert_parity import reference, wav, error


def run(args):
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    Config, Network = reference(args.reference)
    network = Network(Config.from_json_file(str(args.mert_checkpoint/'config.json'))).eval()
    network.load_state_dict(torch.load(args.mert_checkpoint/'pytorch_model.bin',
                           weights_only=True, mmap=True, map_location='cpu'), assign=True)
    if args.mert_dtype == 'f16':
        torch.nn.utils.parametrize.remove_parametrizations(network.encoder.pos_conv_embed.conv,
                                                           'weight', leave_parametrized=True)
        for parameter in network.parameters():
            if parameter.ndim > 1:
                parameter.data = parameter.data.half().float()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.mert_checkpoint)
    paths = list(args.clamp3_checkpoint.glob('*.pth'))
    if len(paths) != 1:
        raise ValueError('CLaMP3 checkpoint must contain exactly one published .pth')
    state = torch.load(paths[0], weights_only=True, mmap=True, map_location='cpu')['model']
    config = BertConfig(vocab_size=1, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072, max_position_embeddings=128)
    with torch.device('meta'):
        audio = BertModel(config)
    audio.load_state_dict({k[len('audio_model.'):]:v for k,v in state.items()
                           if k.startswith('audio_model.')}, assign=True, strict=True)
    audio.embeddings.register_buffer('position_ids', torch.arange(128).expand(1,-1), persistent=False)
    audio.embeddings.register_buffer('token_type_ids', torch.zeros(1,128,dtype=torch.long), persistent=False)
    audio.eval()
    cli = str(args.build/'clap-model')
    subprocess.run([str(args.build/'mert-model-test'), str(args.mert_gguf), str(args.output)], check=True)
    rng = np.random.default_rng(719)
    cases = {'broadband': rng.normal(.1,.12,24000).astype('float32'),
             # Unequal chunk statistics expose per-window rather than recording normalization.
             'two_chunks': np.concatenate([rng.normal(.2,.13,120000),
                                            rng.normal(-.3,.03,24000)]).astype('float32'),
             'resampled': np.frombuffer((args.output/'resampled-control.wav').read_bytes()[44:],
                                        dtype='<f4').copy()}
    report = []
    for name, pcm in cases.items():
        path = args.output/f'{name}.wav'
        wav(path, pcm)
        normalized = processor(pcm, sampling_rate=24000, return_tensors='pt').input_values
        chunks = []
        with torch.no_grad():
            for start in range(0,len(pcm),120000):
                chunk = normalized[:,start:start+120000]
                if chunk.shape[-1] < 24000:
                    break
                states = network(chunk, output_hidden_states=True).hidden_states
                chunks.append(torch.stack(states).mean(-2).mean(0)[0])
            features = torch.stack(chunks)
            x = torch.cat([torch.zeros(1,768), features, torch.zeros(1,768)])[None]
            mask = torch.zeros(1,128,dtype=torch.long); mask[:,:x.shape[1]] = 1
            x = torch.nn.functional.pad(x,(0,0,0,128-x.shape[1]))
            y = audio(inputs_embeds=x, attention_mask=mask).last_hidden_state
            y = (y*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
            expected = torch.nn.functional.linear(y,state['audio_proj.weight'],state['audio_proj.bias'])[0].numpy()
        # For the resampled case the oracle consumes the exact native-resampler control;
        # the actual CLI consumes 48 kHz input, not the pre-resampled waveform.
        actual_path = args.output/'resampled-input.wav' if name == 'resampled' else path
        raw = json.loads(subprocess.check_output([cli,'-m',str(args.mert_gguf),'-a',str(actual_path),
                                                 '--clamp3-features','-t','2']))
        native_features = np.asarray(raw['values'],dtype='float32').reshape(raw['frames'],raw['dimension'])
        error(native_features,features.numpy(),name+'/MERT',0.00002)
        np.save(args.output/'features.npy',native_features)
        file_embedding = json.loads(subprocess.check_output([cli,'-m',str(args.clamp3_gguf),
                '--audio-features',str(args.output/'features.npy'),'-t','2']))
        actual = json.loads(subprocess.check_output([cli,'-m',str(args.clamp3_gguf),
                '--mert-model',str(args.mert_gguf),'-a',str(actual_path),'-t','2']))
        np.testing.assert_allclose(actual,file_embedding,rtol=0,atol=1e-6)
        error(actual,expected,name+'/published-pipeline',0.0005)
        delta = np.asarray(actual)-expected
        report.append(dict(case=name,frames=len(chunks),max_abs=float(np.max(np.abs(delta))),
                           rmse=float(np.sqrt(np.mean(delta**2)))))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Published MERT -> published CLaMP3 pipeline passed',args.mert_dtype,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build',type=Path)
    parser.add_argument('mert_checkpoint',type=Path)
    parser.add_argument('clamp3_checkpoint',type=Path)
    parser.add_argument('reference',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--mert-gguf',type=Path,required=True)
    parser.add_argument('--clamp3-gguf',type=Path,required=True)
    parser.add_argument('--mert-dtype',choices=['f32','f16'],default='f32')
    run(parser.parse_args())
