"""Public C image API validation and CLI surface with generated fixtures."""
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile

from image import png
from tiny_model import write_model


def main():
    lib=C.CDLL(sys.argv[1])
    lib.clap_create.argtypes=[C.c_char_p,C.c_void_p,C.POINTER(C.c_void_p)]
    lib.clap_create.restype=C.c_int
    lib.clap_destroy.argtypes=[C.c_void_p]
    lib.clap_encode_image.argtypes=[C.c_void_p,C.POINTER(C.c_ubyte),C.c_size_t,C.c_int,C.c_int,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
    lib.clap_encode_image.restype=C.c_int
    lib.clap_encode_image_file.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
    lib.clap_encode_image_file.restype=C.c_int
    lib.clap_last_error.restype=C.c_char_p
    with tempfile.TemporaryDirectory(prefix='clap-image-api-') as tmp:
        root=Path(tmp)
        model=root/'tiny.gguf';write_model(model)
        picture=root/'图像.png';picture.write_bytes(png(1,1,bytes([80,100,120])))
        broken=root/'broken.png';broken.write_bytes(b'not an image')
        handle=C.c_void_p()
        assert lib.clap_create(str(model).encode(),None,C.byref(handle))==0
        try:
            rgb=(C.c_ubyte*3)(80,100,120)
            def call(function,*args,status):
                output=C.c_void_p(123);count=C.c_size_t(123)
                actual=function(*args,C.byref(output),C.byref(count))
                assert actual==status,(actual,lib.clap_last_error())
                assert output.value is None and count.value==0
                return lib.clap_last_error().decode()
            call(lib.clap_encode_image,None,rgb,3,1,1,status=1)
            for data,size,w,h in [(None,3,1,1),(rgb,2,1,1),(rgb,3,0,1),(rgb,3,-1,1),(rgb,3,1,0),(rgb,3,2147483647,2147483647)]:
                call(lib.clap_encode_image,handle,data,size,w,h,status=1)
            assert 'image' in call(lib.clap_encode_image,handle,rgb,3,1,1,status=3).lower()
            for path in [None,b'']:
                call(lib.clap_encode_image_file,handle,path,status=1)
            call(lib.clap_encode_image_file,handle,str(root/'missing.png').encode(),status=3)
            call(lib.clap_encode_image_file,handle,str(broken).encode(),status=3)
            assert 'image' in call(lib.clap_encode_image_file,handle,str(picture).encode(),status=3).lower()
        finally:
            lib.clap_destroy(handle)
        if len(sys.argv)>2:
            cli=sys.argv[2]
            help_result=subprocess.run([cli,'embedding','--help'],capture_output=True,text=True,timeout=30)
            assert help_result.returncode==0 and '--image' in help_result.stdout
            base=[cli,'embedding','--model',str(model),'--image',str(picture)]
            result=subprocess.run(base,capture_output=True,text=True,timeout=30)
            assert result.returncode!=0 and 'image' in result.stderr.lower(),result.stderr
            for extra in [['--text','hello'],['--frames'],['--crop-offset','1']]:
                result=subprocess.run(base+extra,capture_output=True,text=True,timeout=30)
                assert result.returncode!=0,result.stdout
            result=subprocess.run([cli,'embedding','--model',str(model),'--image',str(broken)],capture_output=True,text=True,timeout=30)
            assert result.returncode!=0 and ('decode' in result.stderr.lower() or 'image' in result.stderr.lower())
    print('C image API validation/output reset and CLI image-option checks passed')


if __name__=='__main__':
    main()
