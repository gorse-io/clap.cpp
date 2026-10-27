# clap.cpp Python bindings (CPU)

Build and install from a recursive checkout:

```sh
git clone --recursive https://github.com/gorse-io/clap.cpp.git
cd clap.cpp
python -m pip install ./python
```

Requires Python 3.9+, a C++17 compiler and CMake 3.20+. The build backend installs
CMake/Ninja if necessary. NumPy is the only runtime Python dependency; PyTorch,
FFmpeg, a separate `clap` executable and GPU drivers are not required.
The extension statically links the CPU runtime and bundled ICU. GPU backends and
OpenMP are disabled. x86 wheels use baseline instructions rather than the build
machine's instruction set. No pretrained weights are downloaded automatically.

## Usage

Convert a supported checkpoint to GGUF with the repository's conversion tools,
then load it once and reuse it:

```python
from clap_cpp import CLAP

with CLAP("model.gguf", threads=4) as model:
    text_embedding = model.encode_text("A dog barking")
    audio_embedding = model.encode_audio("dog.wav")
    print(model.projection_dimension, model.audio_sample_rate)
```

`CLAP` uses the same GGUF architecture dispatch as the native library (CLAP,
CLaMP3, MERT-v1, MERT-v2, MuQ and MuQ-MuLan). An operation unsupported by a
particular architecture raises a native error; for example audio-only encoders
cannot embed text. CLaMP3 requires a GGUF with its bundled MERT frontend.

## API

- `CLAP(path, threads=0)`: `str`/`pathlib.Path`; zero selects the native thread
  default. `close()` is idempotent and the context manager closes on exit.
- `model.encode_text(text_or_tokens)`: one UTF-8 string or token-ID sequence to a
  one-dimensional float32 NumPy embedding.
- `model.tokenize(text, max_length=512)`: native tokenizer returning a list of
  integer IDs. Its limit must be compatible with the loaded model; strings may
  be truncated by the tokenizer. Prefer `encode_text(str)` for normal use.
- `model.encode_audio(pcm, sample_rate=48000, crop_offset=0)`: mono waveform to a float32
  embedding. Automatic model-rate resampling; crop offset uses the native
  model-rate sample units and is not supported by every architecture.
- `model.audio_frames(pcm, sample_rate=48000)`: native frame representations as a
  `[frames, dimension]` float32 array, for architectures exposing frames.
- `model.clamp3_features(pcm, sample_rate=24000)`: MERT-v1 CLaMP3 preprocessing
  features as a `[windows, dimension]` float32 array (at least one second).
- `model.projection_dimension`, `model.audio_sample_rate`: read-only integers.
- `model.encode_audio(path, crop_offset=0)`: `str`/`pathlib.Path` to a WAV, MP3
  or FLAC file. Decodes internally, averages channels to mono and automatically
  resamples from the file's sample rate to the model's rate. Crop offset has the
  same model-rate sample units as the PCM overload.

PCM accepts one-dimensional numeric arrays or sequences. Noncontiguous arrays
and other numeric dtypes are converted to contiguous float32; the binding copies
inputs before releasing the GIL, so inference never retains a Python buffer.
Empty, nonfinite or multidimensional PCM and nonpositive sample rates are
rejected. Returned arrays own their storage and survive model closure.
Native invalid arguments map to `ValueError`, other native runtime errors to
`RuntimeError`. Calls after closing raise `RuntimeError`.

The binding releases the GIL for inference/loading/decoding/resampling and
serializes all operations on each model (including closing). Different model
instances can execute concurrently. Embedding normalization is architecture
specific and identical to the C++ API; the binding does not normalize outputs.

## Development and source distributions

```sh
python -m pip install './python[test]' build
python -m pytest python/tests
python -m build python
```

The source distribution includes the pinned native sources and third-party
licenses; a wheel can be rebuilt from that archive without a Git checkout or
CMake-time downloads. Build dependencies must still be available from the Python
package index or a local wheelhouse. Wheel publishing and GPU support are out of
scope for this initial binding.
