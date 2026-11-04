# clap.cpp

[![CI](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml)

C++20/ggml implementations of audio embedding inference features:

- Supports WAV, MP3, and FLAC audio input.
- Automatically resamples audio to the model's required.
- Automatically mixes stereo and multichannel audio down to mono.

## Supported models

| Model | Text | Audio | Embedding dimensions | Sample rate |
| --- | --- | --- | --- | --- |
| [CLAP](https://huggingface.co/laion/larger_clap_music) | ✅ | ✅ | 512 | 48 kHz |
| [CLaMP3](https://github.com/sanderwood/clamp3) | ✅ | ✅ | 768 | 24 kHz |
| [MERT v1](https://huggingface.co/m-a-p/MERT-v1-95M) | ❌ | ✅ | 768 | 24 kHz |
| [MERT v2 FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) / [30s](https://huggingface.co/m-a-p/MERT-v2-30s) | ❌ | ✅ | 1,024 | 24 kHz |
| [MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) | ❌ | ✅ | 1,024 | 24 kHz |
| [MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | ✅ | ✅ | 512 | 24 kHz |

GGUF models are available at [Audio Embedding GGUF](https://huggingface.co/collections/gorse-io/audio-embedding-gguf) collection.

## Bindings

| Binding | CPU | CUDA | Vulkan | Install |
| --- | --- | --- | --- | --- |
| [Python](python) | ✅ | ❌ | ❌ | `pip3 install clap-cpp` |
| [Rust](rust) | ✅ | ❌ | ❌ | `cargo install clap-cpp` |
| [Node.js](node) | ✅ | ❌ | ❌ | `npm install clap-cpp` |

## C API

`include/clap.h` is a standalone C11 header with an opaque `CLAP` handle and
`clap_*` functions (also callable from C++ via `extern "C"`). The CLI, HTTP
service, and Python, Node.js and Rust bindings all use this C API; there is no
parallel C++ `CLAP` interface. The implementation still requires a C++20 compiler.
The language bindings' user-facing APIs are unchanged.

```c
#include <clap.h>
#include <stdio.h>

int main(void) {
    CLAP *model = NULL;
    float *embedding = NULL;
    size_t count = 0;
    clap_runtime_options options = clap_default_options();
    options.device = "cpu";
    if (clap_create("model.gguf", &options, &model) != CLAP_OK) {
        fprintf(stderr, "%s\n", clap_last_error());
        return 1;
    }
    clap_status status = clap_encode_text(model, "A jazz quartet.", &embedding, &count);
    if (status != CLAP_OK)
        fprintf(stderr, "%s\n", clap_last_error());
    /* Use count floats on success. */
    clap_free(embedding);
    clap_destroy(model);
    return status == CLAP_OK ? 0 : 1;
}
```

Link the CMake `clap` target; CMake supplies the native runtime dependencies for
both static and shared builds. C callers do not need C++ headers. Manual static
linking needs the C++ runtime plus ggml and ICU dependencies. Windows DLL callers
define `CLAP_SHARED` (the CMake target propagates it automatically).

The header documents ownership and error handling: each embedding/token buffer
uses `clap_free`, device lists use `clap_free_devices`, and frame outputs use
`clap_free_audio_frames`. Outputs are cleared on failure; do not overwrite an
unfreed output. `clap_last_error` is a thread-local diagnostic, valid until the
next fallible call on that thread. Serialize calls on the same handle. Device
selection, activation precision, tokenization, text/token encoding, mono PCM and
WAV/MP3/FLAC path encoding, frame representations, and model dimensions/rates are
all available without STL types or exceptions crossing the API.

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

## Development

C/C++ formatting uses clang-format 18.1.8 and the repository's `.clang-format`.
Install it and format all tracked project sources (submodules are excluded):

```sh
python -m pip install clang-format==18.1.8
git ls-files -z -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx' | xargs -0 clang-format -i
```

To run the same formatting check as GitHub Actions:

```sh
git ls-files -z -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx' | xargs -0 clang-format --dry-run --Werror
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
Third-party dependencies retain their respective licenses.
