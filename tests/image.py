"""Generated image decoding fixtures; no Pillow, downloads or tracked binaries."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib


def png(width, height, pixels, color=2):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    raw = b''.join(b'\0' + pixels[y*width*channels:(y+1)*width*channels] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, color, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def main():
    executable = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix='clap-image-') as tmp:
        root = Path(tmp)
        rgb = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 50, 100, 150, 0, 0, 0, 255, 255, 255])
        def check(name, data, width, height, pixels):
            path = root / name
            path.write_bytes(data)
            result = subprocess.run([executable, str(path)], capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, (name, result.stderr)
            assert json.loads(result.stdout) == dict(width=width, height=height, pixels=list(pixels)), result.stdout
        check('图像 with spaces.dat', png(3, 2, rgb), 3, 2, rgb)
        rgba = bytes([80, 100, 120, 0, 30, 40, 50, 255])
        check('alpha.png', png(2, 1, rgba, 6), 2, 1, bytes([80, 100, 120, 30, 40, 50]))
        check('gray.png', png(2, 1, bytes([20, 210]), 0), 2, 1, bytes([20]*3+[210]*3))
        check('gray-alpha.png', png(2, 1, bytes([20, 0, 210, 255]), 4), 2, 1, bytes([20]*3+[210]*3))
        # Bottom-up BMP has BGR channels and padded scanlines.
        rows = []
        for y in reversed(range(2)):
            row = b''.join(rgb[(y*3+x)*3:(y*3+x)*3+3][::-1] for x in range(3))
            rows.append(row+b'\0'*3)
        bmp_data=b''.join(rows)
        bmp=(b'BM'+struct.pack('<IHHI', 54+len(bmp_data),0,0,54)
             +struct.pack('<IiiHHIIiiII',40,3,2,1,24,0,len(bmp_data),0,0,0,0)+bmp_data)
        check('rgb.bmp',bmp,3,2,rgb)
        # Top-left TGA, packed BGR.
        tga=struct.pack('<BBBHHBHHHHBB',0,0,2,0,0,0,0,0,3,2,24,32)
        tga+=b''.join(rgb[i:i+3][::-1] for i in range(0,len(rgb),3))
        check('rgb.tga',tga,3,2,rgb)
        # JPEG is intentionally lossy: test uniform image dimensions and color tolerance.
        if len(sys.argv)>2:
            ppm=root/'uniform.ppm'; jpg=root/'uniform.jpg'
            ppm.write_bytes(b'P6\n8 8\n255\n'+bytes([80,120,160])*64)
            subprocess.run([sys.argv[2],'-v','error','-i',str(ppm),'-frames:v','1',str(jpg)],check=True,timeout=30)
            result=subprocess.run([executable,str(jpg)],capture_output=True,text=True,timeout=30)
            assert result.returncode==0,result.stderr
            image=json.loads(result.stdout)
            assert (image['width'],image['height'])==(8,8)
            assert all(abs(v-[80,120,160][i%3])<=4 for i,v in enumerate(image['pixels']))
        for name, data in [('empty.png',b''),('invalid.png',b'not an image'),
                           ('truncated.png',png(3,2,rgb)[:35]),
                           ('oversized.png',png(3,2,rgb)[:16]+struct.pack('>II',65536,65536)+png(3,2,rgb)[24:])]:
            path=root/name;path.write_bytes(data)
            result=subprocess.run([executable,str(path)],capture_output=True,text=True,timeout=30)
            assert result.returncode!=0 and result.stderr,(name,result.stdout)
        oversized_file=root/'oversized-file.png'
        with oversized_file.open('wb') as stream:
            stream.truncate((128 << 20) + 1)
        result=subprocess.run([executable,str(oversized_file)],capture_output=True,text=True,timeout=30)
        assert result.returncode!=0 and '128 MiB' in result.stderr
        result=subprocess.run([executable,str(root/'missing.png')],capture_output=True,text=True,timeout=30)
        assert result.returncode!=0 and result.stderr
    print('PNG/RGBA/gray/BMP/TGA/JPEG decoding and invalid-input checks passed')


if __name__=='__main__':
    main()
