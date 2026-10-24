#!/usr/bin/env python3
"""CLaMP3 waveform CLI; feature/source fallbacks are deliberately removed."""
import json
from pathlib import Path
import subprocess
import sys
import wave
binary,fixture=map(Path,sys.argv[1:3])
path=fixture/'input.wav'
with wave.open(str(path),'wb') as output:
    output.setnchannels(1);output.setsampwidth(2);output.setframerate(48000)
    output.writeframes(b'\0\0'*48000)
result=subprocess.run([str(binary), 'embedding','-m',str(fixture/'model.gguf'),'-a',str(path),'-t','2'],capture_output=True,text=True)
assert result.returncode==0,result.stderr
assert len(json.loads(result.stdout))==6
for args in [['--audio-features','obsolete.npy'],['--mert-model',str(fixture/'model.gguf')],
             ['-a',str(path),'--crop-offset','0']]:
    result=subprocess.run([str(binary), 'embedding','-m',str(fixture/'model.gguf'),*args],capture_output=True,text=True)
    assert result.returncode!=0 and result.stderr,args
result=subprocess.run([str(binary), 'embedding','-m',str(fixture/'bad-architecture.gguf'),'--text','a'],capture_output=True,text=True)
assert result.returncode!=0 and 'architecture' in result.stderr
print('CLaMP3 bundled waveform CLI and obsolete-option rejection passed')
