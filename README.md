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
| [EmbeddingGemma 2](https://huggingface.co/unsloth/embeddinggemma-2-GGUF) | ✅ | ✅ | 768 | 16 kHz |

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

### EmbeddingGemma 2 (text and audio)

Load the published [Unsloth EmbeddingGemma 2 GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF)
files directly; no conversion or Python inference is required. Text needs only
the main model. Audio additionally needs `mmproj-Q8_0.gguf` (or the published
F16/BF16 mmproj), which contains the audio tower and projector. For example:

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

This integration supports text and native waveform embeddings; **`encode_image`
remains unsupported**, even though the published mmproj also contains vision
weights. Audio is encoded directly, not transcribed to text:

```sh
./build/clap embedding --model embeddinggemma-2-UD-Q4_K_XL.gguf \
  --mmproj mmproj-Q8_0.gguf --audio recording.wav
```

```c
CLAP *model = NULL;
float *embedding = NULL;
size_t dimension = 0;
if (clap_create_with_mmproj("embeddinggemma-2-UD-Q4_K_XL.gguf", "mmproj-Q8_0.gguf",
                            NULL, &model) == CLAP_OK) {
    clap_status status = clap_encode_audio_file(model, "recording.wav", 0, &embedding, &dimension);
    /* Check status; on failure clap_last_error() describes the error. */
    clap_free(embedding);
    clap_destroy(model);
}
```

WAV/MP3/FLAC input is downmixed and resampled to 16 kHz. Audio preprocessing
follows the published Gemma4 frontend and uses the first 30 seconds, matching
the default processor's truncation policy. The audio Conformer/projector's
soft tokens pass through the same text backbone and final mean/L2 pooling,
producing a normalized 768D embedding in the joint space. Audio requires the
matching mmproj; text still works without it. Crop offsets and frame APIs are
not supported for this joint encoder. The HTTP endpoint remains text-only.

Use published GGUFs directly; the checkpoint converter and local quantization
presets remain scoped to the six existing audio/joint architectures.

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
