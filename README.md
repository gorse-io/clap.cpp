# clap.cpp

C++17/ggml implementations of HuggingFace `laion/larger_clap_music` and CLaMP3,
with native text and audio-feature inference through `clap-model`. The original `clap` embedding
command and HTTP server intentionally retain their dummy behavior.

## Requirements

- CMake 3.20 or newer
- A C++17 compiler
- ICU source submodule (included by recursive checkout; no system ICU installation required)
- Python 3 (only when building tests; conversion/oracle tests additionally need torch, transformers, numpy)
- FFmpeg with libmp3lame (only when building tests, to generate WAV/MP3/FLAC fixtures)

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
The `clap-cli` target (executable name `clap`) links `CLI11::CLI11` so the
command-line entry point can use its headers without a system installation.

cpp-httplib is included as a submodule under `thirdparty/cpp-httplib`, pinned to
v0.61.0. The `clap-cli` target links `httplib::httplib` for HTTP server/client support.
nlohmann/json is included as a submodule under `thirdparty/json`, pinned to
v3.12.0. Request parsing and response serialization use `nlohmann_json::nlohmann_json`.

ICU is pinned to 78.3 (`21d1eb0f306e1141c10931e914dfc038c06121da`) under
`thirdparty/icu`. By default `cmake/icu.cmake` builds a static Unicode/regex-only
subset directly with CMake, without Autotools, MSBuild project files or generated
locale packages. Unicode property/case tables are compiled from ICU's checked-in
data headers; no ICU shared library or external `.dat` file is needed at runtime.
Unused conversion, normalization, collation, formatting and locale services are
disabled: this is the tokenizer's ICU subset, not a general-purpose ICU SDK.

To use an installed full ICU instead, configure with `-DCLAP_USE_SYSTEM_ICU=ON`
(requires the `uc`/`i18n` development libraries, e.g. `libicu-dev` on Debian/Ubuntu).
The bundled targets stay static even when `BUILD_SHARED_LIBS=ON`.

dr_libs is pinned as a submodule under `thirdparty/dr_libs`. The core library
compiles `dr_wav`, `dr_mp3` and `dr_flac` directly; audio decoding requires no
FFmpeg executable or system codec libraries to build or run the library and
executables. Only the test fixture generation step uses FFmpeg.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

### Vulkan

Enable ggml's native Vulkan backend (no custom shaders or matrix kernels):

```sh
# Debian/Ubuntu build dependencies:
sudo apt-get install libvulkan-dev glslc spirv-headers
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vulkan --config Release --parallel
ctest --test-dir build-vulkan -C Release --output-on-failure
./build-vulkan/clap-model --model model.gguf --text 'A jazz quartet.'
```

The native model's linear layers use the first ggml Vulkan device, with weights
cached on that device. Other operations remain on CPU. Missing devices or
unsupported operations use CPU; set `CLAP_VULKAN=0` to force CPU execution.
To require actual Vulkan execution in the tests rather than accept fallback,
configure with `-DCLAP_VULKAN_TEST_REQUIRE_DEVICE=ON` as well. Local validation used
Mesa's software Vulkan driver to check F32/F16 numerical parity, weight reuse,
explicit CPU selection, and missing-driver fallback.

Published releases additionally build `clap-linux-amd64-vulkan` and
`clap-linux-arm64-vulkan` on Ubuntu 24.04. They require the system Vulkan loader
(`libvulkan1` on Debian/Ubuntu) and a compatible Vulkan ICD/driver; the SDK and
`glslc` are build-time dependencies only. Like the CPU release assets, these
`clap` executables still expose the dummy embedding/server interfaces described
below; real model inference currently uses the separately built `clap-model`.

## Native model layout

The C++ organization follows `llama.cpp/src/models`: shared runtime services are
separate from architecture-specific parameter/tensor loading and encoder code.

`src/CMakeLists.txt` builds one `clap` core library from `CLAP_CORE_SOURCES` and
`models/*.cpp`. Both executables and the model tests link this same core library;
there is no separate models library. The `clap-cli` target keeps its executable
name `clap`, while native model inference uses `clap-model` as before.

| Path | Responsibility |
| --- | --- |
| `include/clap.h` | Public, dependency-light C++ API (`clap::Model`, `decode_audio`, `resample_audio`, `log_mel`) |
| `src/model.h`, `src/model.cpp` | Internal architecture interface, dispatch and public API implementation |
| `src/model-loader.*` | Shared GGUF metadata, mapping lifetime and tensor validation |
| `src/model-ops.*` | Shared ggml matrix operations (CPU/optional Vulkan) and CPU normalization |
| `src/audio.cpp`, `src/audio-decoder.cpp`, `src/resample.cpp`, `src/vocab.*` | Native audio frontend, Unicode byte-level BPE and XLM-R Unigram tokenization |
| `src/models/models.h`, `src/models/*.cpp` | CLAP and CLaMP3 specializations: hyperparameters, tensor contracts, encoders and projections |
| `examples/embedding/embedding.cpp` | Real text/audio inference executable (`clap-model`) |
| `convert_hf_to_gguf.py` | Offline HuggingFace-to-GGUF conversion; no Python at inference |

Convert a local `laion/larger_clap_music` checkpoint containing `config.json`,
`pytorch_model.bin`, `tokenizer.json` and `preprocessor_config.json`, then run:

```sh
python convert_hf_to_gguf.py checkpoint/ model.gguf
./build/clap-model --model model.gguf --text 'A jazz quartet playing saxophone.'
./build/clap-model --model model.gguf --audio music.wav
```

For CLAP, the native executable returns a normalized 512-dimensional JSON array. Audio
accepts WAV (including PCM16, PCM24 and float32), MP3 and FLAC, detected by file
contents rather than extension. Channels are averaged to mono, then automatically
resampled to 48 kHz for inference. `--crop-offset` is measured in samples of the
resampled 48 kHz waveform.

### CLaMP3

CLaMP3 conversion accepts its published nested `.pth` checkpoint. Keep the
XLM-R `config.json` and `tokenizer.json` beside it, or pass an explicit combined
test/custom config with `--clamp3-config` and tokenizer with `--tokenizer`:

```sh
python convert_hf_to_gguf.py \
  checkpoint/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth \
  clamp3-f16.gguf --dtype f16
./build/clap-model --model clamp3-f16.gguf --text 'A piano sonata.'
./build/clap-model --model clamp3-f16.gguf --audio-features mert.npy
```

`mert.npy` must be a C-contiguous little-endian float32 NumPy array shaped
`[frames, 768]` or `[1, frames, 768]`, produced by the official MERT-v1-95M
preprocessing. CLaMP3 does not accept raw waveform input: this project does not
embed a MERT extractor, and `--audio` reports that limitation. The runtime adds
the official leading/trailing zero vectors, uses overlapping final 128-frame
chunks, masked mean pooling, remainder-weighted chunk aggregation, and a linear
projection. Unlike CLAP output, CLaMP3 returns a **768-dimensional**, non-L2-normalized
projection. Text uses the XLM-R Unigram vocabulary and its serialized Precompiled
normalizer, with the tokenizer embedded in GGUF and no runtime network access.
Text longer than a chunk is processed in full rather than truncated at 128/512 tokens.

The converter safely memory-maps the checkpoint, reads a nested `model` state
dictionary, omits the unsupported symbolic encoder, and supports `--dtype f32`
and `--dtype f16`. Symbolic ABC/MTF/MusicXML/image input is not supported.

The public `clap::decode_audio(path)` API returns `clap::Audio` with mono float
`samples` and the original `sample_rate`; decoding itself does not resample or
require 48 kHz. `clap::resample_audio(audio, target_rate)` separately resamples
mono PCM (default: 48 kHz) using a centered Hann-windowed sinc filter with
anti-alias filtering for downsampling. Duration is rounded up to the next output
sample, and same-rate input is unchanged. Neither operation requires an external
codec/resampling library. Empty/unsupported inputs and non-finite samples are rejected;
decoded output is limited to 1 GiB. Synthetic codec fixtures are not stored in
the repository. When tests are enabled, the build runs `tests/audio/generate.py`
with FFmpeg to generate them under `build/tests/audio` before CTest runs.
The build regenerates missing or outdated fixtures automatically. Configure
with `-DBUILD_TESTING=OFF` to build without Python or FFmpeg test dependencies.

Inference defaults to CPU, with optional Vulkan linear layers; fusion CLAP, CLaMP3 symbolic input,
quantization and batching are not supported. The optional CLAP Transformers oracle
suite is enabled with `-DCLAP_MODEL_PARITY=ON` and a Python interpreter containing
`torch`, `transformers` and `numpy`. The deterministic synthetic CLaMP3 oracle
suite uses `-DCLAMP3_MODEL_PARITY=ON`; it checks short and overlapping-long text
and MERT feature sequences in F32/F16 and is not evidence about a published
checkpoint. To compare an actual published checkpoint against Hugging Face
XLM-R/BERT, place exactly one official `.pth` and the XLM-R config/tokenizer in
`checkpoint/`, enable `CLAMP3_MODEL_PARITY`, then run:

```sh
python tests/models/clamp3_checkpoint_parity.py build checkpoint/ build/published-clamp3 --dtype f32
# Reuse an existing conversion instead of writing a second GGUF:
python tests/models/clamp3_checkpoint_parity.py build checkpoint/ build/published-clamp3-f16 \
  --dtype f16 --gguf clamp3-f16.gguf
```

For raw music files, first run `preprocessing/audio/extract_mert.py` in the
official `sanderwood/clamp3` checkout using `--model_path m-a-p/MERT-v1-95M`
and `--mean_features`; pass its resulting `.npy` to `--audio-features`.
Existing dummy interfaces below are unchanged.

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
