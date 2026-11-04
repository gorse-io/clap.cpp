"""Run the strict C11 consumer against generated waveform-capable weights."""
import subprocess
import sys
import tempfile
from pathlib import Path
import wave

from tiny_model import write_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))
from tiny_models import mert_fixture

with tempfile.TemporaryDirectory(prefix="clap-c-api-") as directory:
    root = Path(directory)
    model = root / "model.gguf"
    write_model(model, waveform=True)
    audio = root / "silence.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48000)
        output.writeframes(bytes(4800 * 2))
    mert = root / "mert.gguf"
    mert_fixture(mert)
    subprocess.run([sys.argv[1], str(model), str(audio), str(mert)], check=True)
    if len(sys.argv) > 2:
        import ctypes as c
        library = c.CDLL(sys.argv[2])
        library.clap_last_error.restype = c.c_char_p
        library.clap_create.argtypes = [c.c_char_p, c.c_void_p, c.POINTER(c.c_void_p)]
        library.clap_create.restype = c.c_int
        library.clap_destroy.argtypes = [c.c_void_p]
        library.clap_free.argtypes = [c.c_void_p]
        library.clap_encode_text.argtypes = [c.c_void_p, c.c_char_p,
                                           c.POINTER(c.POINTER(c.c_float)), c.POINTER(c.c_size_t)]
        library.clap_encode_text.restype = c.c_int
        handle = c.c_void_p()
        assert library.clap_create(str(model).encode(), None, c.byref(handle)) == 0
        try:
            reference = write_model(root / "reference.gguf", waveform=True)
            for case in reference["cases"]:
                values, count = c.POINTER(c.c_float)(), c.c_size_t()
                assert library.clap_encode_text(handle, case["input"].encode(), c.byref(values), c.byref(count)) == 0
                try:
                    assert count.value == len(case["embedding"])
                    assert max(abs(values[i] - x) for i, x in enumerate(case["embedding"])) < 3e-5
                finally:
                    library.clap_free(values)
        finally:
            library.clap_destroy(handle)
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        barrier = Barrier(2)

        def thread_error(path):
            output = c.c_void_p()
            status = library.clap_create(path, None, c.byref(output))
            before = library.clap_last_error()
            barrier.wait(timeout=10)
            assert library.clap_last_error() == before
            assert not output.value
            return status, before

        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(thread_error, None)
            b = pool.submit(thread_error, b"/missing/thread-model.gguf")
            invalid, missing = a.result(), b.result()
        assert invalid[0] == 1 and missing[0] == 3
        assert invalid[1] and missing[1] and invalid[1] != missing[1]
        assert library.clap_last_error() == b""
