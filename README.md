# clap.cpp

[![CI](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml)
[![Release](https://github.com/gorse-io/clap.cpp/actions/workflows/release.yml/badge.svg)](https://github.com/gorse-io/clap.cpp/actions/workflows/release.yml)

C++17/ggml implementations of Hugging Face `laion/larger_clap_music`, CLaMP3,
and `m-a-p/MERT-v2-FullSong`, with native inference through `clap-model`. The
original `clap` embedding command and HTTP server intentionally retain their dummy behavior.

## Requirements

- CMake 3.20 or newer
- A C++17 compiler
- ICU source submodule (included by recursive checkout; no system ICU installation required)
- Python 3 (only for model conversion and tests; install Python dependencies with `python -m pip install -r requirements.txt`)
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

miniaudio 0.11.25 (`9634bedb5b5a2ca38c1ee7108a9358a4e233f14d`) is pinned
as a submodule under `thirdparty/miniaudio`. It provides WAV/MP3/FLAC decoding,
mono conversion and resampling. Device I/O, threading, the engine, resource
manager, node graph, encoding and signal generation are disabled.
Audio decoding requires no
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

### CUDA 12.9 release build

The published-release workflow also compiles `clap-linux-amd64-cuda12.9` on
Ubuntu 22.04 using CUDA 12.9 and ggml's native CUDA backend. CPU-only build hosts
are supported: `GGML_NATIVE=OFF` uses ggml's portable CUDA architecture list
instead of detecting a local GPU. No CUDA inference or GPU tests are run.
CUDA runtime and cuBLAS are linked statically (`GGML_STATIC=ON`); GPU execution
still requires a compatible NVIDIA driver. The CUDA compiler and cuBLAS
development packages are build-time dependencies only.

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_STATIC=ON
cmake --build build-cuda --config Release --target clap-cli --parallel 2
```

This adds a CUDA-enabled release build, not new model-backend selection: native
model linear dispatch remains CPU/optional Vulkan as documented above, and the
released `clap` embedding/server interfaces still return dummy embeddings.

### Docker

Initialize the submodules before building. Images include both `clap` and
`clap-model`, and reuse the Linux CMake release presets. `BACKEND` selects
`cpu` (default), `cuda` (CUDA 12.9), or `vulkan`; amd64 and arm64 are supported.
Windows/macOS and the RISC-V/LoongArch cross-build presets are not Docker targets.

```sh
# Docker build args select the backend; BuildKit selects the target architecture.
docker build --build-arg BACKEND=cpu -t clap:cpu .
docker build --build-arg BACKEND=cuda -t clap:cuda .
docker build --build-arg BACKEND=vulkan -t clap:vulkan .

# Bake defaults to CPU. Environment variables configure the default target.
docker buildx bake --load
BACKEND=vulkan BUILD_JOBS=4 docker buildx bake --load
# Alternatively, choose one or more named targets.
docker buildx bake cpu cuda vulkan --load
# Multi-platform builds need native builders or QEMU; push to a registry.
IMAGE=ghcr.io/your-org/clap docker buildx bake cpu \
  --set '*.platform=linux/amd64' --set '*.platform+=linux/arm64' --push
```

`BUILD_JOBS` is optional (CPU defaults to 2, CUDA/Vulkan to 1) and is also accepted
by `docker build --build-arg BUILD_JOBS=4`. Vulkan shader compilation can use
substantial memory; keep one build job on memory-constrained hosts. `IMAGE` sets the Bake image repository
(default `clap`); tags are `cpu`, `cuda`, and `vulkan`. No GPU is needed to compile.
The CUDA presets disable optional NCCL so the image does not depend on a shared
NCCL library from the development image; CUDA runtime and cuBLAS remain static.

```sh
docker run --rm clap:cpu --help
docker run --rm -v "$PWD:/data" --entrypoint clap-model clap:cpu \
  --model /data/model.gguf --text 'A jazz quartet.'
# Requires NVIDIA Container Toolkit and a compatible host NVIDIA driver.
docker run --rm --gpus all clap:cuda --help
```

Vulkan images include the loader, not a GPU driver/ICD. Supply a compatible
vendor driver/ICD and device access for your host (for example `/dev/dri` for
Mesa); passing a device alone does not install its driver in the container.
The CLI/model backend limitations described above also apply to these images.

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
| `src/models/models.h`, `src/models/*.cpp` | CLAP, CLaMP3 and MERT2 specializations: hyperparameters, tensor contracts, encoders and projections |
| `examples/embedding/embedding.cpp` | Real text/audio inference executable (`clap-model`) |
| `convert_hf_to_gguf.py` | Offline HuggingFace-to-GGUF conversion CLI; no Python at inference |
| `conversion/__init__.py` | Architecture-to-module map and lazy model-class lookup |
| `conversion/base.py` | Shared model base, architecture detection, safe checkpoint loading and GGUF serialization |
| `conversion/clap.py`, `conversion/clamp3.py`, `conversion/mert2.py` | Model-specific config validation, tensor mapping and conversion |

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

### Q8_0 and Q4_K_M quantization (CLAP and CLaMP3)

Builds include the native `clap quantize` subcommand. First convert the checkpoint
with the existing Python converter (F32 recommended to avoid double rounding),
then quantize locally with upstream ggml's canonical `ggml_quantize_chunk`:

```sh
python convert_hf_to_gguf.py checkpoint/ model-f32.gguf --dtype f32
./build/clap quantize model-f32.gguf model-q8_0.gguf  # default remains Q8_0
./build/clap quantize --type Q4_K_M model-f32.gguf model-q4_k_m.gguf
# --type Q8_0 is also accepted; preset names are case-insensitive.
./build/clap-model --model model-q8_0.gguf --text 'A jazz quartet.'
./build/clap-model --model model-q8_0.gguf --audio music.wav
# The same quantizer accepts CLaMP3 F32/F16 GGUF; use --audio-features for CLaMP3.
```

Q8_0 is **mixed precision**: tensors with ggml rank greater than one and a
contiguous row width (`ne[0]`, the last PyTorch dimension) divisible by 32 are
quantized. All other tensors retain their original F32/F16 bytes, including
biases, normalization vectors and CLAP's 4x4 patch convolution. Embeddings and
eligible relative-position tables are supported. Q8_0 stores 32 signed 8-bit
values and one FP16 scale per block (34 bytes), not one raw byte per weight.

**Q4_K_M** is a mixed **model preset**, not a tensor type. Eligible tensors
have rank greater than one and contiguous row width divisible by **256**.
The deterministic encoder policy `q4_k_m-v1-all-v-ffn-down-projection-q6_k` is:

- Use canonical **Q6_K** for every eligible attention value matrix
  (`*.attention.self.value.weight`) and FFN down/output matrix
  (`*.output.dense.weight`, excluding `*.attention.output.dense.weight`).
- Also use Q6_K for CLAP's `text_proj.linear{1,2}.weight` and
  `audio_proj.linear{1,2}.weight`, and CLaMP3's `text_proj.weight` and
  `audio_proj.weight`.
- Use canonical **Q4_K** for all other eligible matrices, including attention
  query/key/output, FFN intermediate/up, embeddings and patch merging.
- Preserve original F32/F16 bytes for vectors and all incompatible row widths;
  CLAP's 4x4 convolution is unchanged. There is **no Q8 fallback** for K-ineligible
  tensors. Models without any K-compatible matrices fail rather than silently
  producing a Q8 model. Actual tensor mixtures depend on model dimensions.

This adapts the sensitive value/FFN-down roles in the official
[llama.cpp quantization policy](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp),
not an exact implementation of its decoder policy. Upstream's `use_more_bits`
selects the first/last eighth and every third middle layer for Q6_K, with
architecture-specific exceptions. Here **all** value/down layers and final
embedding projections are protected, without decoder layer numbering, MoE/GQA
rules or an importance matrix. Q4_K uses 144 bytes per 256 weights (4.5 bits/weight)
and Q6_K uses 210 bytes (6.5625 bits/weight); neither is a raw four/six-bit array.
GGUF records model-file type **15** (`LLAMA_FTYPE_MOSTLY_Q4_K_M` in
[llama.h](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)) and
the policy string above in `clap.quantization_policy`. The pinned standalone
ggml enum has no `GGML_FTYPE_MOSTLY_Q4_K_M`; its `MOSTLY_Q4_K = 12` is **not**
the interoperable mixed-model identifier.

The loader keeps Q8/K weights mapped (Windows uses its existing file buffer);
linear layers use native ggml quantized matrix multiplication. Scalar reads
decode only the containing block; embedding hotpaths decode one row at a time,
not a K block per channel. There is no whole-model dequantization.
Graph execution allocates the actual planned scratch size, including activation
packing for large F16/Q8/K matrix operations. Quantization streams row chunks,
preserves architecture/tokenizer metadata, and updates GGUF file-type and
quantization-version metadata. `general.alignment` is intentionally normalized
to the writer's 32-byte alignment, including for inputs using other alignments.
It requires no new Python/runtime dependency.
Only F32/F16 input is accepted: requantization is rejected. The output must not
exist (including dangling symlinks). Temporary files are created exclusively
beside the destination and published with an atomic, no-clobber hard link, so a
concurrently created destination is preserved. The destination filesystem must
support hard links; otherwise conversion fails safely without a rename fallback.
Temporary files are removed on ordinary completion or error; abrupt process
termination may leave one behind. Publication does not promise crash durability.

`clap_quantize` and `clap_quantize_k` (ordinary CTest) check canonical bytes
against ggml reference routines, mixed precision, metadata, scalar/row reads,
native Q4_K/Q6_K matmul, multi-chunk conversion, >2 MiB graph scratch and invalid
inputs. Optional CLAP/CLaMP3 Transformers suites exercise Q8 and Q4_K_M models
from both F32 and F16 sources, using 256-wide fixtures that really contain **both**
K types. Q8 uses an independent Python decoder; K storage values come from a
test-only C++ ggml traits exporter, separate from production scalar/row reads.
The oracle also approximates native Q8_0/Q8_K activation packing; CLAP normalized
embedding absolute tolerance is 0.01 and CLaMP3 raw-projection tolerance is 0.002.
These storage-rounded checks isolate inference correctness from weight error;
they are not quantization-quality benchmarks for published models.

A local `laion/larger_clap_music` F32 smoke converted 180 matrices: the GGUF shrank
from **777,060,800** to **128,926,752 bytes** (about 6.03x smaller). Against the
original Transformers checkpoint, one text prompt gave cosine **0.995159**
(max absolute error **0.013608**); one second of 440 Hz PCM audio gave cosine
**0.989932** (max error **0.022304**). Both native outputs were finite, normalized
512-vectors; the F32 baseline max error was below 6e-8. This measures only a
single prompt/synthetic tone, **not retrieval accuracy or general audio quality**.
Reproduce locally (torch/transformers/numpy required; no network access):

```sh
python tests/models/quantized_checkpoint_smoke.py build checkpoint/ model-f32.gguf \
  model-q4_k_m.gguf build/q4-full-smoke.json
```

No full published CLaMP3 Q4 quality evaluation is claimed; its K inference tests
use synthetic XLM-R/BERT text and MERT-feature fixtures.

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

### MERT-v2-FullSong

The native MERT2 implementation follows the official custom configuration,
model and preprocessor source at pinned Hugging Face revision
[`7001f7657276238629e61c7746ce70735e0604c0`](https://huggingface.co/m-a-p/MERT-v2-FullSong/commit/7001f7657276238629e61c7746ce70735e0604c0).
Review that source before loading it with `trust_remote_code`; conversion itself
only parses JSON and safely reads `model.safetensors`.

Download a local snapshot at that exact revision, then convert it while recording
the full source SHA in GGUF metadata:

```sh
python convert_hf_to_gguf.py checkpoint/ mert-v2-fullsong-f16.gguf \
  --dtype f16 --mert2-revision 7001f7657276238629e61c7746ce70735e0604c0
./build/clap-model --model mert-v2-fullsong-f16.gguf --audio complete-song.flac
./build/clap-model --model mert-v2-fullsong-f16.gguf --audio complete-song.flac --frames
```

Input is mixed to mono and resampled to the checkpoint's 24 kHz rate. The native
frontend implements the checkpoint's centered 2048-point power spectrogram,
128-bin mel projection, power-to-dB conversion and learned fixed normalization;
inference then runs all three ConvNeXt subsampling stages and all 24 bidirectional
rotary Conformer blocks. No Python or runtime network access is required.

Without `--frames`, the CLI mean-pools all final frames and prints one
1,024-dimensional recording embedding, matching the official model-card example.
`--frames` instead prints an object with `frames`, `dimension`, and a flat
time-major `values` array. In C++, `Model::audio_frames()` exposes the same
sequence as `clap::AudioFrames`; `Model::audio()` returns its mean. At published
settings the frame count is `floor(input_samples / 960)` (25 Hz).

FullSong means the complete supplied waveform is processed. There is no implicit
30-second slice, crop, window aggregation, or tail discard beyond the official
frontend/frame-ratio semantics. The published checkpoint declares 360 seconds of
training context; longer input is not silently shortened. Bidirectional attention
has quadratic time/memory cost in the output frame count, so long CPU runs can be
expensive. `--crop-offset` is intentionally rejected for MERT2.

For F16 conversion, matrix and convolution weights use F16 storage while vector
parameters and all official float32 frontend tensors remain F32. The converter
streams safetensors one at a time rather than materializing the 632M-parameter
checkpoint in RAM. Quantized GGUF types, batching, intermediate hidden-state
output, GPU backends and the 30-second MERT2 variant are not supported.

The deterministic tiny official-model suite covers the waveform frontend,
ConvNeXt, Conformer, pooled/frame APIs, F32/F16 storage, CLI, and a 31.25-second
no-truncation case:

```sh
cmake -S . -B build -DMERT2_MODEL_PARITY=ON \
  -DMERT2_REFERENCE_SOURCE=/path/to/reviewed/7001f7657276238629e61c7746ce70735e0604c0
cmake --build build --parallel
ctest --test-dir build -R mert2 --output-on-failure
```

Those random tiny weights are not published-checkpoint evidence. After verifying
`model.safetensors` against the official manifest SHA-256
`e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d`, run the
separate actual-weight oracle:

```sh
python tests/models/mert2_checkpoint_parity.py build checkpoint/ build/published-mert2
# Reuse a prior F32 conversion:
python tests/models/mert2_checkpoint_parity.py build checkpoint/ build/published-mert2 \
  --gguf mert-v2-fullsong-f32.gguf
```

The public `clap::decode_audio(path)` API returns `clap::Audio` with mono float
`samples` and the original `sample_rate`; decoding itself does not resample or
require 48 kHz. `clap::resample_audio(audio, target_rate)` separately resamples
mono PCM (default: 48 kHz) using miniaudio's linear interpolator and eighth-order
low-pass filter (before downsampling or after upsampling). Unlike the previous
centered sinc, this causal filter has startup latency and a frequency-dependent
phase response; very short clips include that startup transient. It is not
bit-identical to the old resampler. Duration is rounded up to the next output
sample, with the last input value extended when fractional output needs lookahead;
same-rate input is unchanged. Neither operation requires an external
codec/resampling library. Empty/unsupported inputs and non-finite samples are rejected;
decoded output is limited to 1 GiB. Synthetic codec fixtures are not stored in
the repository. When tests are enabled, the build runs `tests/audio/generate.py`
with FFmpeg to generate them under `build/tests/audio` before CTest runs.
The build regenerates missing or outdated fixtures automatically. Configure
with `-DBUILD_TESTING=OFF` to build without Python or FFmpeg test dependencies.

Inference uses CPU or optional Vulkan linear layers; fusion CLAP, CLaMP3 symbolic
input and batching are not supported. F32, F16, mixed Q8_0 and mixed Q4_K_M GGUF
inference are supported. The optional CLAP Transformers oracle
suite is enabled with `-DCLAP_MODEL_PARITY=ON` and a Python interpreter containing
`torch`, `transformers` and `numpy`. The deterministic synthetic CLaMP3 oracle
suite uses `-DCLAMP3_MODEL_PARITY=ON`; it checks short and overlapping-long text
and MERT feature sequences in F32/F16/Q8_0/Q4_K_M and is not evidence about a published
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

## License

This project is licensed under the [Apache License 2.0](LICENSE).
Third-party dependencies retain their respective licenses.
