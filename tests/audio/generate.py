"""Generate synthetic codec fixtures into a build directory (requires FFmpeg)."""
import argparse
import math
from pathlib import Path
import shutil
import struct
import subprocess
import wave

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="Directory for generated audio fixtures")
parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable with libmp3lame support")
options = parser.parse_args()
ROOT = options.output.resolve()
ROOT.mkdir(parents=True, exist_ok=True)


def ffmpeg(source, destination, *args):
    subprocess.run([options.ffmpeg, "-v", "error", "-y", "-i", str(ROOT / source),
                    *args, str(ROOT / destination)], check=True)


for channels, name in [(1, "mono"), (2, "stereo")]:
    with wave.open(str(ROOT / (name + ".wav")), "wb") as output:
        output.setparams((channels, 2, 48000, 0, "NONE", "not compressed"))
        output.writeframes(b"".join(
            struct.pack("<h", round(12000 * math.sin(2 * math.pi * frequency * i / 48000)))
            for i in range(4800) for frequency in (440, 880)[:channels]))
    ffmpeg(name + ".wav", name + ".flac")
    ffmpeg(name + ".wav", name + ".mp3", "-codec:a", "libmp3lame", "-b:a", "192k")
ffmpeg("mono.wav", "pcm24.wav", "-c:a", "pcm_s24le")
ffmpeg("mono.wav", "float.wav", "-c:a", "pcm_f32le")
ffmpeg("mono.wav", "44100.flac", "-ar", "44100")
shutil.copyfile(ROOT / "mono.flac", ROOT / "audio.dat")
(ROOT / "invalid.bin").write_bytes(b"not an audio file\n")
with wave.open(str(ROOT / "empty.wav"), "wb") as output:
    output.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
