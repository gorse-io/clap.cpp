"""Exercise the low-RAM published-oracle implementation with tiny weights.

The synthetic branch mode bypasses published identity checking, but uses the
same meta construction, mmap loading, storage rounding and persisted references.
This is not evidence about published checkpoint accuracy.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

build,fixture,source,muq_source=map(Path,sys.argv[1:])
root=Path(__file__).resolve().parents[2]
for name in ('muq','text'):
    shutil.copyfile(fixture/(name+'-config.json'),fixture/(name+'_config.json'))
for dtype in ('f32','f16'):
    out=fixture/('meta-oracle-'+dtype);out.mkdir(exist_ok=True)
    base=[sys.executable,str(root/'tests/models/muq_mulan_checkpoint_parity.py'),
          '--checkpoint',str(fixture),'--gguf',str(fixture/f'model-{dtype}.gguf'),
          '--dtype',dtype,'--cli',str(build/'clap'),'--source',str(source),
          '--muq-source',str(muq_source),'--out',str(out)]
    for branch in ('text','audio'):
        subprocess.run(base+['--branch',branch],check=True)
        cases=json.loads((out/(branch+'-reference.json')).read_text())
        if branch=='audio':cases=[cases]
        metrics=[]
        for index,case in enumerate(cases):
            args=['--text',case['text']] if branch=='text' else ['--audio',case['wave']]
            result=subprocess.run([str(build/'clap'), 'embedding','--model',str(fixture/f'model-{dtype}.gguf'),'--threads','2',*args],check=True,capture_output=True,text=True)
            actual=json.loads(result.stdout);expected=case['expected']
            error=max(abs(a-b) for a,b in zip(actual,expected))
            (out/f'{branch}-{index}-native.json').write_text(json.dumps(actual))
            metrics.append({'case':index,'max_abs_error':error,'dimension':len(actual)})
        (out/(branch+'-metrics.json')).write_text(json.dumps(metrics,indent=2))
        print(dtype,branch,metrics,flush=True)
        assert all(m['dimension']==6 and m['max_abs_error']<1e-4 for m in metrics),metrics
print('Synthetic meta/mmap oracle pipeline passed; published identity checks remain separate')
