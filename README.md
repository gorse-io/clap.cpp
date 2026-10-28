# clap.cpp

[![CI](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml)

C++17/ggml implementations of audio embedding inference features:

- Supports WAV, MP3, and FLAC audio input.
- Automatically resamples audio to the model's required.
- Automatically mixes stereo and multichannel audio down to mono.

## Supported models

| Model | Text | Audio | Embedding dimensions | Sample rate |
| --- | --- | --- | --- | --- |
| [CLAP](https://huggingface.co/laion/larger_clap_music) | ✅ | ✅ | 512 | 48 kHz |
| [CLaMP3](https://github.com/sanderwood/clamp3) | ✅ | ✅ | 768 | 24 kHz |
| [MERT v1](https://huggingface.co/m-a-p/MERT-v1-95M) | ❌ | ✅ | 768 | 24 kHz |
| [MERT v2](https://huggingface.co/m-a-p/MERT-v2-FullSong) | ❌ | ✅ | 1,024 | 24 kHz |
| [MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) | ❌ | ✅ | 1,024 | 24 kHz |
| [MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | ✅ | ✅ | 512 | 24 kHz |

| [EmbeddingGemma 2](https://huggingface.co/unsloth/embeddinggemma-2-GGUF) | ✅ | ❌ | 768 | — |

GGUF models are available at [Audio Embedding GGUF](https://huggingface.co/collections/gorse-io/audio-embedding-gguf) collection.

## Bindings

| Binding | CPU | CUDA | Vulkan | Install |
| --- | --- | --- | --- | --- |
| [Python](python) | ✅ | ❌ | ❌ | `pip3 install clap-cpp` |
| [Rust](rust) | ✅ | ❌ | ❌ | `cargo install clap-cpp` |
| [Node.js](node) | ✅ | ❌ | ❌ | `npm install clap-cpp` |

## Usage

## Embeddings command

Load a converted GGUF and generate real native text or audio embeddings:

```sh
clap embedding --model model.gguf --text 'A jazz quartet.'
clap embedding --model model.gguf --audio music.flac
```

### EmbeddingGemma 2 (text only)

Load the published [Unsloth EmbeddingGemma 2 GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF)
files directly; no conversion, Python inference, or `mmproj` is required. For example,
download `embeddinggemma-2-UD-Q4_K_XL.gguf` and run:

```sh
./build/clap embedding --model embeddinggemma-2-UD-Q4_K_XL.gguf --text 'Hello world'
./build/clap embedding --model embeddinggemma-2-UD-Q4_K_XL.gguf \
  --text 'task: search result | query: How do plants grow?'
./build/clap embedding --model embeddinggemma-2-UD-Q4_K_XL.gguf \
  --text 'title: none | text: Plants use sunlight for photosynthesis.'
./build/clap serve --model embeddinggemma-2-UD-Q4_K_XL.gguf
```

`encode_text` returns a normalized 768-dimensional embedding. Both BOS and EOS
are added automatically; mean pooling includes these tokens and any prompt.
Retrieval prefixes are **not** added automatically: supply
`task: search result | query: ` for queries and `title: none | text: ` for
untitled documents, or `title: {title} | text: {content}` for titled documents.
The tokenizer preserves Unicode, case, and repeated whitespace. Google advertises
an 8192-token context; use that limit even though the GGUF positional capacity is
larger. Native computation preserves F32 activations with F16/BF16 storage and
accepts the published Q4_K/Q5_K/Q6_K/Q8_0 mixed tensor types.

This integration is **text only**: no `encode_image`, image projector, or audio
encoding is implemented for this architecture. Use the published GGUF directly;
the checkpoint converter and local quantization presets below remain scoped to
the six existing audio/joint architectures.

## Embeddings server

Start the server with a text-capable GGUF, such as CLAP, CLaMP3 or MuQ-MuLan
(default listen address: `127.0.0.1:8080`):

```sh
clap serve --model model.gguf --host 127.0.0.1 --port 8080
```

The model is loaded once at startup. Its API identifier is the GGUF **filename**
(`model.gguf` in this example), not its full path or a Hugging Face repository ID.
Send an OpenAI-compatible text embeddings request:

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"model.gguf","input":["hello","world"],"encoding_format":"float"}'
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
Third-party dependencies retain their respective licenses.