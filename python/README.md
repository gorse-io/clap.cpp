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
