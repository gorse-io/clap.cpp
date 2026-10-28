# clap.cpp Python bindings

## Install

```sh
pip install clap-cpp
```

## Usage

```python
from clap_cpp import CLAP

with CLAP("model.gguf", threads=4) as model:
    text_embedding = model.encode_text("A dog barking")
    audio_embedding = model.encode_audio("dog.wav")
    print(model.projection_dimension, model.audio_sample_rate)
```

For EmbeddingGemma 2 audio, also load the matching audio mmproj GGUF:

```python
with CLAP("embeddinggemma2.gguf", threads=4, mmproj="mmproj-Q8_0.gguf") as model:
    audio_embedding = model.encode_audio("recording.wav")
```

`mmproj` accepts a string or `pathlib.Path`. Text still works without it; the
existing positional `threads` argument is unchanged. Missing/incompatible
projectors raise native errors. Image encoding is not supported.
