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

## Embeddings server

Start the server with a GGUF supporting the requested modality, such as CLAP,
CLaMP3 or MuQ-MuLan (default listen address: `127.0.0.1:8080`):

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

Audio inputs use one `input_audio` content block per item. Encode the complete
WAV, MP3 or FLAC file as standard base64 (no whitespace or data-URL prefix):

```sh
python3 -c 'import base64,json,sys; print(json.dumps({"input":[{"content":[{"type":"input_audio","input_audio":{"data":base64.b64encode(open(sys.argv[1],"rb").read()).decode()}}]}]}))' music.flac |
  curl http://127.0.0.1:8080/v1/embeddings \
    -H 'Content-Type: application/json' --data-binary @-
```

`input_audio.format` is optional: the decoder detects the codec from the bytes.
Text strings and audio objects can share the same `input` array; results retain
input order. Each audio item must contain **exactly one** audio block; combining
multiple blocks or pooling text and audio within one item is not supported.
Audio is decoded in memory and downmixed to mono; native model preprocessing
handles its sample rate just as `clap embedding --audio` does.

Requests are limited to **32 MiB of JSON**, including base64 overhead, and 1024
items. Larger bodies receive HTTP 413. Malformed input/base64 or undecodable audio
receives a JSON HTTP 400 error; model inference failures receive JSON HTTP 500.
Responses containing audio omit `usage` rather than inventing token counts; text-only
requests retain their existing token-usage behavior. Output encoding is `float`.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
Third-party dependencies retain their respective licenses.
