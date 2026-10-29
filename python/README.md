# clap.cpp Python bindings

## Install

```sh
pip install clap-cpp
```

## Type hints

The package includes type stubs and a `py.typed` marker for IDE completion and
static checking with mypy or Pyright. `encode_text` and `encode_audio` return
`list[float]`; `audio_frames` and `clamp3_features` return time-major
`list[list[float]]`. PCM input accepts mono numeric sequences such as lists or
tuples. The runtime has no NumPy dependency. Text, token-sequence, audio-file and
PCM input overloads are declared separately.

## Usage

```python
from clap_cpp import CLAP

with CLAP("model.gguf", threads=4) as model:
    text_embedding = model.encode_text("A dog barking")
    audio_embedding = model.encode_audio("dog.wav")
    print(model.projection_dimension, model.audio_sample_rate)
```
