# clap.cpp

<img width=160 src="assets/clap.png"/>

[![CI](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml)
[![GitHub Release](https://img.shields.io/github/v/release/gorse-io/clap.cpp)](https://github.com/gorse-io/clap.cpp/releases/latest)

C++20/ggml implementations of audio embedding inference features:

- Supports WAV, MP3, and FLAC audio input.
- Automatically resamples audio to the model's required.
- Automatically mixes stereo and multichannel audio down to mono.

## Supported models

| Model | Text | Audio | Embedding dimensions | Sample rate |
| --- | --- | --- | --- | --- |
| [CLAP](https://huggingface.co/laion/larger_clap_music) | ✅ | ✅ | 512 | 48 kHz |
| [CLAP2 Large](https://huggingface.co/laion/clap2-large-audioxl-s1B) | ✅ | ✅ | 960 | 32 kHz |
| [CLAP2 Medium](https://huggingface.co/laion/clap2-medium-audioxl-s1B) | ✅ | ✅ | 640 | 32 kHz |
| [CLAP2 Small](https://huggingface.co/laion/clap2-small-audioxl-s1B) | ✅ | ✅ | 512 | 32 kHz |
| [CLAP2 XS](https://huggingface.co/laion/clap2-xs-audioxl-s1B) | ✅ | ✅ | 320 | 32 kHz |
| [CLAP2 XXS](https://huggingface.co/laion/clap2-xxs-audioxl-s1B) | ✅ | ✅ | 224 | 32 kHz |
| [CLaMP3](https://github.com/sanderwood/clamp3) | ✅ | ✅ | 768 | 24 kHz |
| [MERT v1 95M](https://huggingface.co/m-a-p/MERT-v1-95M) | ❌ | ✅ | 768 | 24 kHz |
| [MERT v1 330M](https://huggingface.co/m-a-p/MERT-v1-330M) | ❌ | ✅ | 1,024 | 24 kHz |
| [MERT v2 FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) / [30s](https://huggingface.co/m-a-p/MERT-v2-30s) | ❌ | ✅ | 1,024 | 24 kHz |
| [MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) | ❌ | ✅ | 1,024 | 24 kHz |
| [MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | ✅ | ✅ | 512 | 24 kHz |

GGUF models are available at [Audio Embedding GGUF](https://huggingface.co/collections/gorse-io/audio-embedding-gguf) collection.

## Bindings

| Binding | CPU | CUDA | Vulkan | Install |
| --- | --- | --- | --- | --- |
| [Python](python) | ✅ | ✅ | ✅ | `pip3 install clap-cpp` |
| [Rust](rust) | ✅ | ✅ | ✅ | `cargo install clap-cpp` |
| [Node.js](node) | ✅ | ❌ | ❌ | `npm install clap-cpp` |

## Usage

### Embeddings command

Load a converted GGUF and generate real native text or audio embeddings:

```sh
clap embedding --model larger_clap_general-q4_k_m.gguf --text 'Sound of a dog'
clap embedding --model larger_clap_general-q4_k_m.gguf --audio dog.wav
```

### Embeddings server

Start the server with a GGUF supporting the requested modality, such as CLAP,
CLaMP3 or MuQ-MuLan (default listen address: `127.0.0.1:8080`):

```sh
clap serve --model larger_clap_general-q4_k_m.gguf --host 127.0.0.1 --port 8080
```

The model is loaded once at startup. Its API identifier is the GGUF **filename**
(`larger_clap_general-q4_k_m.gguf` in this example), not its full path or a Hugging Face repository ID.
Send an OpenAI-compatible text embeddings request:

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"larger_clap_general-q4_k_m.gguf","input":["Sound of a dog"],"encoding_format":"float"}'
```

Audio inputs use one `input_audio` content block per item. Encode the complete
WAV, MP3 or FLAC file as standard base64 (no whitespace or data-URL prefix):

```bash
audio_base64=$(base64 < dog.wav | tr -d '\r\n')
printf '{"input":[{"content":[{"type":"input_audio","input_audio":{"data":"%s"}}]}]}' "$audio_base64" |
  curl http://127.0.0.1:8080/v1/embeddings \
    -H 'Content-Type: application/json' --data-binary @-
```

### Audio search (brute force)

Search the `esc5` audio directory recursively for `dog` and return the ten most related audio files:

```sh
clap search esc5 --model larger_clap_general-q4_k_m.gguf --text 'dog' --top-k 10
```

<details>
<summary>Show example search results</summary>

```json
[
  {"path": "esc5/2-118072-A-0-dog.wav", "score": 0.3736357720143104},
  {"path": "esc5/1-110389-A-0-dog.wav", "score": 0.36165991257288893},
  {"path": "esc5/4-183992-A-0-dog.wav", "score": 0.3574270305924374},
  {"path": "esc5/1-30226-A-0-dog.wav", "score": 0.34910229736736315},
  {"path": "esc5/3-180977-A-0-dog.wav", "score": 0.34359234068457495},
  {"path": "esc5/4-194754-A-0-dog.wav", "score": 0.3425423428503198},
  {"path": "esc5/2-118964-A-0-dog.wav", "score": 0.3389233902904601},
  {"path": "esc5/1-100032-A-0-dog.wav", "score": 0.33009392183878733},
  {"path": "esc5/5-9032-A-0-dog.wav", "score": 0.3280081351893563},
  {"path": "esc5/4-182395-A-0-dog.wav", "score": 0.32210240115509925}
]
```

</details>

## License

This project is licensed under the [Apache License 2.0](LICENSE).
Third-party dependencies retain their respective licenses.
