# clap.cpp

C++17/ggml implementation of HuggingFace `laion/larger_clap_music`, with native
text and audio inference through `clap-model`. The original `clap` embedding
command and HTTP server intentionally retain their dummy behavior.

## Requirements

- CMake 3.20 or newer
- A C++17 compiler
- ICU development headers and libraries (`libicu-dev` on Debian/Ubuntu)
- Python 3 (only when building tests; conversion/oracle tests additionally need torch, transformers, numpy)

## Clone

```sh
git clone --recurse-submodules https://github.com/gorse-io/clap.cpp.git
cd clap.cpp
```

For an existing checkout:

```sh
git submodule update --init --recursive
```

The `ggml/` submodule uses upstream `ggml-org/ggml`, pinned to a commit.
Like stable-diffusion.cpp, it is built with `add_subdirectory` and linked via
its `ggml` CMake target; no system ggml installation is required.

CLI11 is included as a submodule under `thirdparty/CLI11`, pinned to v2.7.2.
The `clap` target links `CLI11::CLI11` so the command-line entry point can use
its headers without a system installation.

cpp-httplib is included as a submodule under `thirdparty/cpp-httplib`, pinned to
v0.61.0. The `clap` target links `httplib::httplib` for HTTP server/client support.
nlohmann/json is included as a submodule under `thirdparty/json`, pinned to
v3.12.0. Request parsing and response serialization use `nlohmann_json::nlohmann_json`.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

## Native model layout

The C++ organization follows `llama.cpp/src/models`: shared runtime services are
separate from architecture-specific parameter/tensor loading and encoder code.

| Path | Responsibility |
| --- | --- |
| `include/clap.h` | Public, dependency-light C++ API (`clap::Model`, `log_mel`) |
| `src/model.h`, `src/model.cpp` | Internal architecture interface, dispatch and public API implementation |
| `src/model-loader.*` | Shared GGUF metadata, mapping lifetime and tensor validation |
| `src/model-ops.*` | Shared CPU ggml matrix operations and normalization |
| `src/audio.cpp`, `src/vocab.*` | Native audio frontend and Unicode byte-level BPE |
| `src/models/models.h`, `src/models/clap.cpp` | CLAP specialization: hyperparameters, tensor contract, HTSAT/RoBERTa encoders and projections |
| `examples/embedding/embedding.cpp` | Real text/audio inference executable (`clap-model`) |
| `convert_hf_to_gguf.py` | Offline HuggingFace-to-GGUF conversion; no Python at inference |

Convert a local `laion/larger_clap_music` checkpoint containing `config.json`,
`pytorch_model.bin`, `tokenizer.json` and `preprocessor_config.json`, then run:

```sh
python convert_hf_to_gguf.py checkpoint/ model.gguf
./build/clap-model --model model.gguf --text 'A jazz quartet playing saxophone.'
./build/clap-model --model model.gguf --audio music.wav
```

The native executable returns a normalized 512-dimensional JSON array. Audio
must already be mono 48 kHz PCM16/float32 WAV. Inference is CPU-only; fusion CLAP,
quantization and batching are not supported. The optional Transformers oracle
suite is enabled with `-DCLAP_MODEL_PARITY=ON` and a Python interpreter containing
`torch`, `transformers` and `numpy`. Existing dummy interfaces below are unchanged.

## Embedding command

Generate dummy embeddings directly, without starting a server or loading a model:

```sh
./build/clap embedding -p "Hello World!"
# 1.0000000 0.0000000 0.0000000

./build/clap embedding -f prompts.txt --embd-output-format json
./build/clap embedding -p 'hello<#sep#>world' \
  --embd-separator '<#sep#>' --embd-output-format array
# [[1.0,0.0,0.0],[1.0,0.0,0.0]]
```

The input and output options follow llama.cpp's `llama-embedding` example:

- Exactly one of `-p/--prompt` or `-f/--file` is required. Input must not be empty.
- `--embd-separator` splits input into prompts (default: an actual newline).
  Empty segments, including a trailing segment, each receive an embedding.
  The separator must not be empty.
- `--embd-output-format` accepts `raw` (default, one space-separated vector per
  line), `array` (an array of vectors), `json` (`object: "list"` with indexed
  embedding objects), or `json+` (also includes `cosineSimilarity` for multiple
  prompts). Structured output is written to stdout; errors go to stderr.

Every prompt receives the fixed vector `[1.0, 0.0, 0.0]`, just like the server.
All cosine similarities are therefore `1.0`. This is only a subset of the
llama.cpp CLI: model loading, tokenization, inference, pooling, and normalization
options are not implemented. No model argument is needed.

## Embeddings server

Start the server (defaults to `127.0.0.1:8080`):

```sh
./build/clap serve --host 127.0.0.1 --port 8080
```

Send a request following the llama.cpp `/v1/embeddings` JSON response shape:

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"dummy","input":["hello","world"],"encoding_format":"float"}'
```

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]},
    {"object": "embedding", "index": 1, "embedding": [1.0, 0.0, 0.0]}
  ],
  "model": "dummy",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

This is a stub: every input gets the same fixed 3-dimensional vector; no model
is loaded and no tokenization or inference takes place. Usage counts are zero.
`input` accepts one non-empty string or an array of 1–1024 non-empty strings.
`model` is optional (defaults to `dummy`) and is echoed in the response.
Only `encoding_format: "float"` (the default) and `dimensions: 3` are supported;
token IDs, multimodal inputs and base64 encoding are not implemented.
Invalid JSON or unsupported parameter values return HTTP 400 with a JSON
`error` object. Request bodies are limited to 1 MiB.

Running `clap` without a subcommand still prints `clap.cpp`.
The executable is located at `build/clap` for single-configuration generators or
`build/Release/clap.exe` for Visual Studio Release builds.

To skip tests, configure with `-DBUILD_TESTING=OFF`.
Tests include a ggml context/tensor smoke test to verify headers and linking.
Dependencies are static by default. `GGML_NATIVE` defaults to `OFF` for portable
builds; configure with `-DGGML_NATIVE=ON` to optimize for the build machine.
Other backend options can be passed directly, for example `-DGGML_CUDA=ON`
(requires the CUDA toolkit).

## Install

```sh
cmake --install build --config Release --prefix ./install
```

The executable is installed under `bin/` in the selected prefix.
