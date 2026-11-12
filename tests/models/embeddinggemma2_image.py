"""Stdlib-only generated Gemma4 vision GGUF and independent scalar oracle.

Tiny configurable pixel budget keeps default tests fast; published metadata uses
HF's 280-soft-token default. F32 activations, storage-rounded F32/F16 weights.
"""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib
import embeddinggemma2 as text


def fixture(storage='f32', optional=True):
    meta = {'general.architecture': 'clip', 'general.type': 'mmproj',
            'clip.has_vision_encoder': True, 'clip.has_audio_encoder': False,
            # Disabled modalities may retain configuration after tensor stripping.
            'clip.audio.projector_type': 'gemma4a',
            'clip.vision.projector_type': 'gemma4v', 'clip.vision.projection_dim': 8,
            'clip.vision.embedding_length': 16, 'clip.vision.feed_forward_length': 23,
            'clip.vision.block_count': 2, 'clip.vision.attention.head_count': 2,
            'clip.vision.patch_size': 2, 'clip.vision.image_size': 18,
            'clip.vision.image_max_pixels': 324,
            'clip.vision.projector.scale_factor': 3,
            'clip.vision.attention.layer_norm_epsilon': 1e-6}
    tensors = {}
    def put(name, shape, norm=False, value=None):
        serial = len(tensors) + 1
        values = [value if value is not None else 0.9 + 0.15*math.sin(i+serial) if norm else
                  0.16*math.sin((i+serial*11)*.31)+.03*math.cos(i*.17) for i in range(math.prod(shape))]
        fmt = 'e' if storage == 'f16' and len(shape) == 2 else 'f'
        values = list(struct.unpack('<'+fmt*len(values), struct.pack('<'+fmt*len(values), *values)))
        tensors[name] = (shape, values)
    put('v.patch_embd.weight', [2,2,3,16])
    put('v.position_embd.weight', [16,64,2])
    put('mm.input_projection.weight', [16,8])
    put('v.std_bias', [16])
    put('v.std_scale', [16], norm=True)
    for layer in range(2):
        p=f'v.blk.{layer}.'
        for n in ['ln1','ln2','attn_post_norm','ffn_post_norm']:
            put(p+n+'.weight',[16],norm=True)
        for n in ['attn_q_norm','attn_k_norm']: put(p+n+'.weight',[8],norm=True)
        for name,ni,no in [('attn_q',16,16),('attn_k',16,16),('attn_v',16,16),('attn_out',16,16),
                           ('ffn_gate',16,23),('ffn_up',16,23),('ffn_down',23,16)]:
            put(p+name+'.weight',[ni,no])
            # Nontrivial clipping verifies both activation and output clamps.
            for suffix,value in [('input_min',-1.1),('input_max',1.2),('output_min',-.9),('output_max',1.0)]:
                put(p+name+'.'+suffix,[1],value=value)
    for suffix,value in [('input_min',-.8),('input_max',.9),('output_min',-.7),('output_max',.8)]:
        put('mm.input_projection.'+suffix,[1],value=value)
    if not optional:
        tensors={name:value for name,value in tensors.items() if name not in ['v.std_bias','v.std_scale'] and
                 not name.endswith(('.input_min','.input_max','.output_min','.output_max'))}
    return meta,tensors


def resize(image, width, height):
    """Separable antialiased Keys bicubic (a=-.5), uint8 image semantics."""
    def weights(src,dst):
        scale=src/dst; support=max(1.,scale)
        def cubic(x):
            x=abs(x)
            return (1.5*x-2.5)*x*x+1 if x<1 else ((-.5*x+2.5)*x-4)*x+2 if x<2 else 0.
        result=[]
        for j in range(dst):
            center=(j+.5)*scale
            w=[(i,cubic((i+.5-center)/support)) for i in range(max(0,int(center-2*support+.5)),min(src,int(center+2*support+.5)))]
            total=sum(v for _,v in w);result.append([(i,v/total) for i,v in w])
        maximum=max(v for row in result for _,v in row)
        precision=0
        while precision<22 and int(.5+maximum*(1<<(precision+1)))<(1<<15):precision+=1
        factor=1<<precision
        return [[(i,int(v*factor+(.5 if v>=0 else -.5))) for i,v in row] for row in result],precision
    if width==len(image[0]) and height==len(image): return image
    (wx,px),(wy,py)=weights(len(image[0]),width),weights(len(image),height)
    # ATen uint8 path quantizes normalized coefficients to int16 independently
    # per axis; each separable pass saturates with half-up integer rounding.
    tmp=[[[max(0,min(255,(sum(row[i][c]*v for i,v in w)+(1<<(px-1)))>>px)) for c in range(3)] for w in wx] for row in image]
    return [[[max(0,min(255,(sum(tmp[i][x][c]*v for i,v in w)+(1<<(py-1)))>>py)) for c in range(3)] for x in range(width)] for w in wy]


def oracle(width,height,pixels,tensors):
    p,k,budget=2,3,9
    hidden=tensors['v.patch_embd.weight'][0][-1];hd=hidden//2
    factor=math.sqrt(budget*k*k*p*p/(width*height))
    h=int(height*factor//(p*k))*(p*k);w=int(width*factor//(p*k))*(p*k)
    if not h: h=p*k;w=min(width//height*p*k,budget*p*k)
    elif not w:w=p*k;h=min(height//width*p*k,budget*p*k)
    image=[[pixels[(y*width+x)*3:(y*width+x+1)*3] for x in range(width)] for y in range(height)]
    image=resize(image,w,h); nx,ny=w//p,h//p
    def weights(n):return tensors[n][1]
    def rms(rows,name=None,dim=None):
        if dim is None:dim=len(rows[0])
        scales=weights(name+'.weight') if name else [1.]*dim
        out=[]
        for row in rows:
            result=[]
            for off in range(0,len(row),dim):
                part=row[off:off+dim];den=math.sqrt(sum(v*v for v in part)/dim+struct.unpack('<f',struct.pack('<f',1e-6))[0])
                result += [v/den*s for v,s in zip(part,scales)]
            out.append(result)
        return out
    def linear(rows,name):
        (ni,no),ww=tensors[name+'.weight']
        def clamp(v,kind):
            low=weights(name+'.'+kind+'_min')[0] if name+'.'+kind+'_min' in tensors else -math.inf
            high=weights(name+'.'+kind+'_max')[0] if name+'.'+kind+'_max' in tensors else math.inf
            return max(low,min(high,v))
        return [[clamp(sum(clamp(v,'input')*ww[j*ni+i] for i,v in enumerate(row)),'output') for j in range(no)] for row in rows]
    def add(a,b):return [[x+y for x,y in zip(u,v)] for u,v in zip(a,b)]
    patch=weights('v.patch_embd.weight');pos=weights('v.position_embd.weight')
    x=[]
    for py in range(ny):
        for px in range(nx):
            x.append([sum((image[py*p+dy][px*p+dx][c]/255*2-1)*patch[((d*3+c)*p+dy)*p+dx]
                          for dy in range(p) for dx in range(p) for c in range(3)) + pos[px*hidden+d]+pos[(64+py)*hidden+d] for d in range(hidden)])
    for layer in range(2):
        root=f'v.blk.{layer}.';normalized=rms(x,root+'ln1')
        q=rms(linear(normalized,root+'attn_q'),root+'attn_q_norm',hd)
        kk=rms(linear(normalized,root+'attn_k'),root+'attn_k_norm',hd)
        v=rms(linear(normalized,root+'attn_v'),None,hd)
        # Independently rotate both halves of each head at theta=100.
        for m in (q,kk):
            for row,values in enumerate(m):
                for head in range(2):
                    for axis,position in enumerate((row%nx,row//nx)):
                        for channel in range(hd//4):
                            angle=position*100**(-4*channel/hd)
                            a=head*hd+axis*(hd//2)+channel;b=a+hd//4
                            values[a],values[b]=values[a]*math.cos(angle)-values[b]*math.sin(angle),values[a]*math.sin(angle)+values[b]*math.cos(angle)
        attended=[]
        for query in q:
            result=[]
            for head in range(2):
                scores=[sum(query[head*hd+d]*key[head*hd+d] for d in range(hd)) for key in kk]
                prob=[math.exp(s-max(scores)) for s in scores];total=sum(prob)
                result += [sum(prob[j]*v[j][head*hd+d] for j in range(len(v)))/total for d in range(hd)]
            attended.append(result)
        x=add(x,rms(linear(attended,root+'attn_out'),root+'attn_post_norm'))
        normalized=rms(x,root+'ln2');gate=linear(normalized,root+'ffn_gate');up=linear(normalized,root+'ffn_up')
        gated=[[.5*a*(1+math.tanh(math.sqrt(2/math.pi)*(a+.044715*a**3)))*b for a,b in zip(u,v)] for u,v in zip(gate,up)]
        x=add(x,rms(linear(gated,root+'ffn_down'),root+'ffn_post_norm'))
    pooled=[]
    for y in range(ny//k):
        for xx in range(nx//k):
            values=[sum(x[(y*k+dy)*nx+xx*k+dx][d] for dy in range(k) for dx in range(k))/k**2*math.sqrt(hidden) for d in range(hidden)]
            if 'v.std_bias' in tensors:values=[(v-b)*s for v,b,s in zip(values,weights('v.std_bias'),weights('v.std_scale'))]
            pooled.append(values)
    return linear(rms(pooled),'mm.input_projection')


def generate(root, binary=None, cli=None):
    root.mkdir(parents=True,exist_ok=True)
    for storage,optional in [('f32',True),('f16',True),('f32',False),('f16',False)]:
        meta,tensors=fixture(storage,optional);cases=[]
        stem=storage+('-optional' if optional else '-plain')
        for width,height in [(18,18),(8,8),(40,20),(20,40),(1,100)]:
            pixels=[(i*37+i//7*19)%256 for i in range(width*height*3)]
            soft=oracle(width,height,pixels,tensors)
            tm,tt,expected=text.fixture(storage,audio_soft=soft)
            tm['tokenizer.ggml.tokens'][-2:]=['<|image>','<image|>']
            image_path = root / f'{width}x{height}.png'
            def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
            raw=b''.join(b'\0'+bytes(pixels[y*width*3:(y+1)*width*3]) for y in range(height))
            image_path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
            cases.append(dict(width=width,height=height,pixels=pixels,path=str(image_path),soft=sum(soft,[]),embedding=expected['audio_embedding']))
        model=root/(stem+'-text.gguf');mmproj=root/(stem+'-vision.gguf');gold=root/(stem+'.json')
        text.save(model,tm,tt,storage);text.save(mmproj,meta,tensors,storage);gold.write_text(json.dumps({'cases':cases}))
        if binary:subprocess.run([binary,str(model),str(mmproj),str(gold)],check=True,timeout=60)
        if cli:
            for case in cases:
                result=subprocess.run([cli,'embedding','--model',str(model),'--mmproj',str(mmproj),
                                       '--threads','2','--image',case['path']],check=True,capture_output=True,text=True,timeout=30)
                actual=json.loads(result.stdout)
                assert len(actual)==len(case['embedding'])
                assert max(abs(a-b) for a,b in zip(actual,case['embedding']))<1e-5
    if binary:
        for name in ['v.unknown.weight','mm.unknown.weight']:
            broken=copy.deepcopy(tensors);broken[name]=([1],[1.])
            path=root/'unknown.gguf';text.save(path,meta,broken,storage)
            subprocess.run([binary,str(model),str(path),str(gold),name],check=True)
        for field,value in [('clip.vision.attention.head_count',3),('clip.vision.patch_size',0),
                            ('clip.vision.attention.layer_norm_epsilon',-1.)]:
            bad=dict(meta);bad[field]=value;path=root/'bad-meta.gguf';text.save(path,bad,tensors,storage)
            subprocess.run([binary,str(model),str(path),str(gold),'vision'],check=True)
        for flag in ['clip.has_vision_encoder','clip.has_audio_encoder']:
            bad=dict(meta);bad[flag]=1;path=root/'bad-flag.gguf';text.save(path,bad,tensors,storage)
            subprocess.run([binary,str(model),str(path),str(gold),flag],check=True)
        import embeddinggemma2_audio as audio
        am,at=audio.fixture(storage,projection=8)
        am['clip.has_vision_encoder']=False
        am['clip.vision.projector_type']='gemma4v'
        tm,tt,_=text.fixture(storage,audio_soft=[[.1]*8])
        audio_model=root/'audio-text.gguf';audio_mm=root/'audio-only.gguf'
        text.save(audio_model,tm,tt,storage);text.save(audio_mm,am,at,storage)
        subprocess.run([binary,str(audio_model),str(audio_mm),'unused','--audio-only'],check=True)
    print('F32/F16 image tower and composed embeddings verified' if binary else 'Generated image fixtures')


if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='--output':generate(Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory(prefix='gemma-image-') as temp:generate(Path(temp),sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
