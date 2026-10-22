"""Offline HF cache fixtures for mandatory CLaMP3/MERT conversion tests."""
import json
import os
from pathlib import Path
import shutil

REVISION = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'

def install(cache, source):
    snapshot = Path(cache)/'models--m-a-p--MERT-v1-95M'/'snapshots'/REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ['config.json', 'preprocessor_config.json', 'pytorch_model.bin']:
        shutil.copyfile(Path(source)/name, snapshot/name)
    os.environ['HF_HUB_CACHE'] = str(Path(cache).resolve())
    os.environ['HF_HUB_OFFLINE'] = '1'
    return snapshot

def tiny(cache, width):
    import torch
    from transformers import HubertConfig, HubertModel
    source = Path(cache)/'source'; source.mkdir(parents=True, exist_ok=True)
    config = HubertConfig(hidden_size=width, intermediate_size=width*2, num_hidden_layers=1,
        num_attention_heads=2, conv_dim=[8]*7, conv_kernel=[10,3,3,3,3,2,2],
        conv_stride=[5,2,2,2,2,2,2], num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
        conv_bias=False, feat_extract_norm='group', hidden_dropout=0., attention_dropout=0.,
        feat_proj_dropout=0., layerdrop=0., mask_time_prob=0., mask_feature_prob=0.)
    with torch.random.fork_rng():
        torch.manual_seed(719)
        network = HubertModel(config).eval()
    raw=config.to_dict();raw.update(model_type='mert_model',sample_rate=24000,
        feat_proj_layer_norm=True, feature_extractor_cqt=False,deepnorm=False)
    (source/'config.json').write_text(json.dumps(raw))
    (source/'preprocessor_config.json').write_text(json.dumps(dict(
        feature_extractor_type='Wav2Vec2FeatureExtractor',feature_size=1,sampling_rate=24000,
        do_normalize=True,padding_value=0,padding_side='right',return_attention_mask=True)))
    torch.save(network.state_dict(),source/'pytorch_model.bin')
    return install(cache,source)
