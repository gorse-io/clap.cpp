"""Shared native/converter adversarial MuQ metadata probes (stdlib only)."""
import copy


def malformed(base):
    for path in [('encoder_dim',), ('encoder_depth',), ('conv_dim',), ('hop_length',),
                 ('n_mels',), ('label_rate',), ('w2v2_config', 'num_attention_heads'),
                 ('w2v2_config', 'intermediate_size'), ('w2v2_config', 'conv_depthwise_kernel_size')]:
        for value in (True, 1.5, 4294967297, 1e100, -1, 0, 2**64-1):
            raw = copy.deepcopy(base)
            obj = raw if len(path) == 1 else raw[path[0]]
            obj[path[-1]] = value
            yield '.'.join(path) + '=' + repr(value), raw
    for key,value in [('n_mels',128.0),('hop_length',240.0),('label_rate',25.0),('encoder_depth',257)]:
        raw=copy.deepcopy(base);raw[key]=value
        yield key+'='+repr(value),raw
    for path in [('stat', 'melspec_2048_mean'), ('stat', 'melspec_2048_std'),
                 ('w2v2_config', 'layer_norm_eps'), ('w2v2_config', 'rotary_embedding_base')]:
        for value in (True, '1', 10**400, 1e100, 1e-100, float('inf'), float('nan')):
            raw = copy.deepcopy(base); raw[path[0]][path[1]] = value
            yield '.'.join(path) + '=' + repr(value), raw
    # Individually bounded dimensions whose tensor products or layer totals
    # exceed signed-int indexing/resource limits. Never generate their schemas.
    for updates in ({'encoder_dim': 65536}, {'conv_dim': 65536},
                    {'encoder_dim': 8192, 'conv_dim': 8192},
                    {'encoder_dim': 8192, 'encoder_depth': 256}):
        raw = copy.deepcopy(base); raw.update(updates)
        yield 'derived product ' + repr(updates), raw


def published(base):
    raw = copy.deepcopy(base)
    raw.update(encoder_dim=1024, encoder_depth=12, conv_dim=512)
    raw['w2v2_config'].update(num_attention_heads=16, intermediate_size=4096,
                              conv_depthwise_kernel_size=31)
    return raw
