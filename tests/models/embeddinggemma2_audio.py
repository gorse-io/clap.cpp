"""Tiny Gemma4-audio GGUF and independent stdlib scalar/DFT oracle.

No model downloads, numpy, torch, or production feature-extractor reuse.
Run: python embeddinggemma2_audio.py --test /path/to/audio-test
The fixture's second conv has two channels, detecting channel/frequency order.
Attention inputs cross chunk boundaries and include future-perturbation cases.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def string(x):
    x = x.encode()
    return struct.pack('<Q', len(x)) + x


def typed(x):
    if isinstance(x, str): return 8, string(x)
    if isinstance(x, bool): return 7, struct.pack('<?', x)
    if isinstance(x, int):
        if -(1 << 31) <= x < (1 << 31): return 5, struct.pack('<i', x)
        if x < (1 << 63): return 11, struct.pack('<q', x)
        return 10, struct.pack('<Q', x)
    return 6, struct.pack('<f', x)


def save(path, metadata, tensors, storage):
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        kind, data = typed(value)
        header += string(key) + struct.pack('<I', kind) + data
    blob = bytearray()
    for name, (shape, values) in tensors.items():
        dtype = 1 if storage == 'f16' and len(shape) == 2 else 0
        raw = struct.pack('<' + ('e' if dtype else 'f') * len(values), *values)
        header += string(name) + struct.pack('<I', len(shape))
        header += struct.pack('<' + 'Q' * len(shape), *shape) + struct.pack('<IQ', dtype, len(blob))
        blob.extend(raw)
        blob.extend(b'\0' * (-len(blob) % 32))
    path.write_bytes(header + b'\0' * (-len(header) % 32) + blob)


def fixture(storage="f32", projection=512):
    hidden, heads, layers, channels, output = 8, 2, 2, [4, 2], 12
    meta = {'general.architecture': 'clip', 'general.type': 'mmproj',
            'clip.has_audio_encoder': True, 'clip.has_vision_encoder': True,
            'clip.audio.projector_type': 'gemma4a', 'clip.audio.projection_dim': projection,
            'clip.audio.embedding_length': hidden, 'clip.audio.feed_forward_length': 4 * hidden,
            'clip.audio.block_count': layers, 'clip.audio.attention.head_count': heads,
            'clip.audio.num_mel_bins': 128, 'clip.audio.attention.layer_norm_epsilon': 1e-6}
    tensors = {}

    def put(name, shape, norm=False, value=None):
        n = math.prod(shape)
        serial = len(tensors) + 1
        values = [value if value is not None else
                  0.9 + 0.15 * math.sin(i + serial) if norm else
                  0.17 * math.sin((i + 11 * serial) * 0.31) + 0.03 * math.cos(i * 0.17)
                  for i in range(n)]
        fmt = 'e' if storage == 'f16' and len(shape) == 2 else 'f'
        values = list(struct.unpack('<' + fmt * n, struct.pack('<' + fmt * n, *values)))
        tensors[name] = (shape, values)

    for i, (inc, outc) in enumerate(zip([1, channels[0]], channels)):
        put(f'a.conv1d.{i}.weight', [3, 3, inc, outc])
        put(f'a.conv1d.{i}.norm.weight', [outc], norm=True)
    put('a.input_projection.weight', [32 * channels[1], hidden])
    put('a.pre_encode.out.weight', [hidden, output])
    put('a.pre_encode.out.bias', [output])
    put('mm.a.input_projection.weight', [output, projection])
    put('mm.a.soft_emb_norm.weight', [output], norm=True)
    # A mixed vision/audio mmproj must be accepted without touching vision tensors.
    put('v.unrelated_fixture.weight', [1], value=0.4)
    for il in range(layers):
        p = f'a.blk.{il}.'
        for name in ['ffn_norm', 'ffn_post_norm', 'ffn_norm_1', 'ffn_post_norm_1',
                     'norm_conv', 'conv_norm', 'ln2', 'attn_pre_norm', 'attn_post_norm']:
            put(p + name + '.weight', [hidden], norm=True)
        put(p + 'conv_dw.weight', [5, hidden])
        # This weight is already softplus-transformed by the GGUF converter.
        put(p + 'per_dim_scale.weight', [hidden // heads], value=0.73)
        for name, ni, no in [('ffn_up', hidden, 4*hidden), ('ffn_down', 4*hidden, hidden),
                             ('ffn_up_1', hidden, 4*hidden), ('ffn_down_1', 4*hidden, hidden),
                             ('attn_q', hidden, hidden), ('attn_k', hidden, hidden),
                             ('attn_v', hidden, hidden), ('attn_out', hidden, hidden),
                             ('conv_pw1', hidden, 2*hidden), ('conv_pw2', hidden, hidden)]:
            put(p + name + '.weight', [ni, no])
            for suffix, value in [('input_min', -.35), ('input_max', .4),
                                  ('output_min', -.3), ('output_max', .32)]:
                put(p + name + '.' + suffix, [1], value=value)
        put(p + 'attn_k_rel.weight', [hidden, hidden])
    return meta, tensors


def frontend(pcm):
    # Direct DFT independently checks the shared FFT and float window rounding.
    frames = (len(pcm) - 161) // 160 + 1
    edges = [700 * (10 ** ((math.log10(1 + 8000 / 700) * i / 129)) - 1) for i in range(130)]
    result = []
    for frame in range(frames):
        samples = [f32(f32(pcm[idx]) * f32(0.5 - 0.5 * math.cos(2*math.pi*n/320)))
                   if 0 <= (idx := frame*160+n-160) < len(pcm) else 0.0 for n in range(320)]
        mag = []
        for k in range(257):
            re = sum(v * math.cos(2*math.pi*k*n/512) for n,v in enumerate(samples))
            im = -sum(v * math.sin(2*math.pi*k*n/512) for n,v in enumerate(samples))
            mag.append(math.hypot(re, im))
        result.append([f32(math.log(0.001 + sum(mag[k] * max(0, min(
            (k*16000/512-edges[b])/(edges[b+1]-edges[b]),
            (edges[b+2]-k*16000/512)/(edges[b+2]-edges[b+1]))) for k in range(257))))
                       for b in range(128)])
    return result


def oracle(features, tensors):
    def weight(n): return tensors[n][1]

    def norm(x, name, layer=False):
        w=weight(name+'.weight') if name+'.weight' in tensors else [1.0]*len(x[0])
        y=[]
        for row in x:
            mean=sum(row)/len(row) if layer else 0
            var=sum((v-mean)**2 for v in row)/len(row)
            y.append([(v-mean)/math.sqrt(var+f32(1e-6))*g for v,g in zip(row,w)])
        return y

    def linear(x,name,clipped=False,bias=False):
        shape,w=tensors[name+'.weight']; ni,no=shape
        if clipped:
            low,high=weight(name+'.input_min')[0],weight(name+'.input_max')[0]
            x=[[max(low,min(high,v)) for v in row] for row in x]
        y=[[sum(row[i]*w[j*ni+i] for i in range(ni)) + (weight(name+'.bias')[j] if bias else 0)
            for j in range(no)] for row in x]
        if clipped:
            low,high=weight(name+'.output_min')[0],weight(name+'.output_max')[0]
            y=[[max(low,min(high,v)) for v in row] for row in y]
        return y

    def residual(x,y,scale=1.0): return [[a+scale*b for a,b in zip(u,v)] for u,v in zip(x,y)]
    def silu(x): return [[v/(1+math.exp(-v)) for v in row] for row in x]

    # Tensor indexing independent of native im2col/ggml implementation.
    x=[[[v] for v in row] for row in features]
    for layer in range(2):
        p=f'a.conv1d.{layer}'; shape,w=tensors[p+'.weight']; _,_,inc,outc=shape
        nt,nf=(len(x)+1)//2,(len(x[0])+1)//2
        y=[]
        for t in range(nt):
            yr=[]
            for f in range(nf):
                cell=[]
                for oc in range(outc):
                    val=0
                    for kt in range(3):
                        for kf in range(3):
                            it,jf=2*t+kt-1,2*f+kf-1
                            if 0<=it<len(x) and 0<=jf<len(x[0]):
                                for ic in range(inc):
                                    val += x[it][jf][ic]*w[((oc*inc+ic)*3+kt)*3+kf]
                    cell.append(val)
                yr.append([max(0,v) for v in norm([cell],p+'.norm',layer=True)[0]])
            y.append(yr)
        x=y
    x=linear([[v for cell in row for v in cell] for row in x],'a.input_projection')
    dim=len(x[0]); hd=dim//2
    pos=[[math.sin(t*math.exp(-math.log(10000)*i/(dim//2-1))) for i in range(dim//2)] +
         [math.cos(t*math.exp(-math.log(10000)*i/(dim//2-1))) for i in range(dim//2)] for t in range(12,-1,-1)]
    for il in range(2):
        p=f'a.blk.{il}.'
        def ff(x,suffix):
            y=norm(x,p+'ffn_norm'+suffix)
            y=linear(silu(linear(y,p+'ffn_up'+suffix,True)),p+'ffn_down'+suffix,True)
            return residual(x,norm(y,p+'ffn_post_norm'+suffix),.5)
        x=ff(x,'')
        z=norm(x,p+'attn_pre_norm')
        q=linear(z,p+'attn_q',True); k=linear(z,p+'attn_k',True); v=linear(z,p+'attn_v',True)
        q=[[a*(hd**-.5)/math.log(2)*weight(p+'per_dim_scale.weight')[j%hd] for j,a in enumerate(row)] for row in q]
        k=[[a*math.log(1+math.e)/math.log(2) for a in row] for row in k]
        rp=linear(pos,p+'attn_k_rel')
        attn=[]
        for t in range(len(x)):
            row=[]; block=t//12; qi=t%12
            for h in range(2):
                # HF _rel_shift: flatten [C,S+1], trim, reshape [C,S].
                # Position index for (qi,s) is (qi*S+s)%(S+1),
                # and source query index is floor((qi*S+s)/(S+1)).
                logits=[]
                for s in range(24):
                    key=block*12-12+s
                    srcq=(qi*24+s)//25; ri=(qi*24+s)%25
                    content=sum(q[t][h*hd+d]*k[key][h*hd+d] for d in range(hd)) if 0<=key<len(x) else 0
                    src_t=block*12+srcq
                    relative=sum(q[src_t][h*hd+d]*rp[ri][h*hd+d] for d in range(hd)) if ri<13 and src_t<len(x) else 0
                    valid=0<=key<len(x) and 0<=t-key<12
                    logits.append(50*math.tanh((content+relative)/50) if valid else -1e9)
                m=max(logits); probs=[math.exp(l-m) for l in logits]; total=sum(probs)
                row.extend(sum(probs[s]/total*v[block*12-12+s][h*hd+d]
                               for s in range(24) if 0<=block*12-12+s<len(x)) for d in range(hd))
            attn.append(row)
        x=residual(x,norm(linear(attn,p+'attn_out',True),p+'attn_post_norm'))
        # GGUF conversion swaps these two norm names relative to HF roles.
        z=linear(norm(x,p+'conv_norm'),p+'conv_pw1',True)
        z=[[row[d]/(1+math.exp(-row[d+dim])) for d in range(dim)] for row in z]
        dw=weight(p+'conv_dw.weight')
        y=[[sum(z[t+j-4][d]*dw[d*5+j] for j in range(5) if t+j>=4) for d in range(dim)] for t in range(len(z))]
        x=residual(x,linear(silu(norm(y,p+'norm_conv')),p+'conv_pw2',True))
        x=norm(ff(x,'_1'),p+'ln2')
    x=linear(x,'a.pre_encode.out',bias=True)
    return linear(norm(x,'mm.a.soft_emb_norm'),'mm.a.input_projection')


def run(output, binary=None):
    output.mkdir(parents=True,exist_ok=True)
    for storage,projection in [('f32',512),('f16',512),('f32',8)]:
        label=storage if projection==512 else f'{storage}-projection{projection}'
        meta,tensors=fixture(storage,projection)
        path=output/f'audio-{label}.gguf'; save(path,meta,tensors,storage)
        front=[]
        for n in [161,321,1601]:
            pcm=[f32(.11*math.sin(i*.173)+.07*math.cos(i*.091)) for i in range(n)]
            features=frontend(pcm)
            front.append(dict(pcm=pcm,features=features,encoded=oracle(features,tensors)))
        tower=[]
        for n in [45,53,101]:
            features=[[f32(.3*math.sin(t*.51+b*.13)+.12*math.cos(t*.017*b)) for b in range(128)] for t in range(n)]
            tower.append(dict(features=features,encoded=oracle(features,tensors)))
        data=output/f'audio-{label}.json'; data.write_text(json.dumps(dict(frontend=front,tower=tower,tolerance=4e-5,boundaries=label=='f32')))
        if binary: subprocess.run([str(binary),str(path),str(data)],check=True)
        # Published EmbeddingGemma2 omits this optional scale entirely.
        plain_tensors=copy.deepcopy(tensors)
        del plain_tensors['mm.a.soft_emb_norm.weight']
        plain_path=output/f'audio-{label}-unweighted.gguf'
        save(plain_path,meta,plain_tensors,storage)
        plain_data=output/f'audio-{label}-unweighted.json'
        plain_data.write_text(json.dumps(dict(frontend=[],tower=[dict(features=tower[-1]['features'],
            encoded=oracle(tower[-1]['features'],plain_tensors))],tolerance=4e-5)))
        if binary: subprocess.run([str(binary),str(plain_path),str(plain_data)],check=True)
    if binary:
        mutations=[('variant',lambda m,t:m.update({'clip.audio.projector_type':'gemma4ua'}),'projector'),
                   ('signed-dimension-overflow',lambda m,t:m.update({'clip.audio.embedding_length':(1<<32)+8}),'dimension'),
                   ('negative-dimension-overflow',lambda m,t:m.update({'clip.audio.embedding_length':-(1<<32)+8}),'dimension'),
                   ('unsigned-dimension-overflow',lambda m,t:m.update({'clip.audio.embedding_length':(1<<64)-1}),'integer'),
                   ('head',lambda m,t:m.update({'clip.audio.attention.head_count':3}),'head'),
                   ('mel',lambda m,t:m.update({'clip.audio.num_mel_bins':64}),'mel'),
                   ('nan-epsilon',lambda m,t:m.update({'clip.audio.attention.layer_norm_epsilon':float('nan')}),'finite'),
                   ('no-clip',lambda m,t:t.pop('a.blk.0.attn_q.input_min'),'missing tensor'),
                   ('bad-clip',lambda m,t:t.update({'a.blk.0.attn_q.input_min':([1],[2.0])}),'clamp'),
                   ('bad-scale',lambda m,t:t.update({'a.blk.0.per_dim_scale.weight':([4],[-1]*4)}),'scale'),
                   ('bad-depthwise',lambda m,t:t.update({'a.blk.0.conv_dw.weight':([4,8],[.1]*32)}),'shape'),
                   ('unknown-audio',lambda m,t:t.update({'a.unknown.weight':([1],[0.0])}),'unrecognized'),
                   ('unknown-audio-projector',lambda m,t:t.update({'mm.a.unknown.weight':([1],[0.0])}),'unrecognized'),
                   ('missing-projector',lambda m,t:t.pop('mm.a.input_projection.weight'),'missing tensor')]
        for name,mutate,message in mutations:
            meta,tensors=fixture('f32'); mutate(meta,tensors)
            path=output/f'invalid-{name}.gguf';save(path,meta,tensors,'f32')
            subprocess.run([str(binary),str(path),'unused',message],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--test',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output: run(args.output,args.test)
    else:
        with tempfile.TemporaryDirectory(prefix='gemma-audio-') as tmp: run(Path(tmp),args.test)
