# clap.cpp

[![CI](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/gorse-io/clap.cpp/actions/workflows/ci.yml)
[![Release](https://github.com/gorse-io/clap.cpp/actions/workflows/release.yml/badge.svg)](https://github.com/gorse-io/clap.cpp/actions/workflows/release.yml)

C++17/ggml implementations of Hugging Face `laion/larger_clap_music`, CLaMP3,
`m-a-p/MERT-v1-95M`, `m-a-p/MERT-v2-FullSong`, OpenMuQ/MuQ and MuQ-MuLan,
with native inference through `clap-model`. The
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

## Python bindings (CPU)

Install from a recursive checkout with `python -m pip install ./python`.
The [`python/` package](python/README.md) provides native text/audio embeddings,
NumPy waveform input and output, audio decoding/resampling, and model lifetime
management. It builds a self-contained CPU extension; no PyTorch or GPU runtime
is needed for inference. See the package README for API examples and tests.

## Node.js bindings (CPU)

Install from a recursive checkout with `npm install ./node`.
The [`node/` package](node/README.md) provides native text/audio embeddings,
`Float32Array` waveform input and output, frame representations, audio
decoding/resampling, and TypeScript declarations. It builds a self-contained CPU
addon. The synchronous API blocks the calling thread; use a worker thread for
server applications. See the package README for API examples and tests.

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

Windows amd64 and arm64 releases use CUDA 13.4.3 installed with Scoop
(`cuda@13.4.3`). Their `clap-windows-*-cuda13.4.zip` assets contain `clap.exe`
and the matching cuBLAS/cuBLASLt runtime DLLs. The Windows presets use Ninja
and select `nvcc` from the Scoop toolkit; Linux CUDA releases remain on CUDA
12.9. CUDA and ROCm release jobs use sccache with Cloudflare R2 storage rather
than the GitHub Actions cache. The ARM64 CUDA preset uses MSVC (required by
the Windows CUDA host toolchain), enables ggml's native CPU feature probing,
and explicitly sets GPU architectures so a build host does not need an NVIDIA GPU.
All CUDA release presets emit Hopper (`90`) as real and virtual code, retaining
PTX while avoiding sccache 0.18.0's PTX-only `compute_90` parsing issue.

This adds a CUDA-enabled release build, not new model-backend selection: native
model linear dispatch remains CPU/optional Vulkan as documented above, and the
released `clap` embedding/server interfaces still return dummy embeddings.

### ROCm amd64 release build

The release workflow builds `clap-linux-amd64-rocm` on Ubuntu 22.04 with
ROCm 6.4.3 and ggml's native HIP backend. The release preset explicitly targets
`gfx908`, `gfx90a`, `gfx942`, `gfx1030`, `gfx1100`, `gfx1101`, `gfx1200`, and
`gfx1201`, so compilation does not require a GPU on the build host.

Install AMD's ROCm 6.4.3 packages `hip-dev`, `hipblas-dev`, `rocblas-dev`, and
`rocm-device-libs`, then:

```sh
cmake --preset linux-amd64-rocm
cmake --build build --config Release --target clap-cli --parallel 1
```

The Linux executable requires the ROCm 6.4.3 HIP, hipBLAS, and rocBLAS shared
libraries under `/opt/rocm/lib` at runtime; they are not bundled.

On Windows amd64, `clap-windows-amd64-rocm.exe` is built on Windows Server 2022
with AMD HIP SDK 6.4.2 (the official 25.Q3 installer). Its Ninja preset uses the
SDK's Clang compilers and explicitly targets `gfx1030`, `gfx1100`, `gfx1101`,
`gfx1102`, `gfx1151`, `gfx1200`, and `gfx1201`, without detecting a local GPU.
To build, use an x64 Visual Studio developer shell with Ninja available, set
`ROCM_PATH` to `C:/Program Files/AMD/ROCm/6.4`, and configure with
`cmake --preset windows-amd64-rocm`. To run the release executable, install
[AMD HIP SDK 6.4.2](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
including its HIP, hipBLAS, and rocBLAS runtime libraries, and add
`C:\Program Files\AMD\ROCm\6.4\bin` to `PATH`. The DLLs and rocBLAS library data
are not bundled; retain the SDK's installed directory layout.

ggml does not support static ROCm linking. GPU execution also requires a
compatible AMD GPU and driver.
This adds a release build, not HIP model dispatch: native model linear layers
remain CPU/optional Vulkan, and the released `clap` embedding/server interfaces
still return dummy embeddings. No ROCm inference or GPU tests are run.

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
| `src/audio.cpp`, `src/vocab.*` | Native audio frontend, Unicode byte-level BPE and XLM-R Unigram tokenization |
| `src/models/models.h`, `src/models/*.cpp` | CLAP, CLaMP3, MERT-v1, MERT2, MuQ and MuQ-MuLan specializations: hyperparameters, tensor contracts, encoders and projections |
| `examples/embedding/embedding.cpp` | Real text/audio inference executable (`clap-model`) |
| `convert_hf_to_gguf.py` | Offline HuggingFace-to-GGUF conversion CLI; no Python at inference |
| `conversion/__init__.py` | Architecture-to-module map and lazy model-class lookup |
| `conversion/base.py` | Shared model base, architecture detection, safe checkpoint loading and GGUF serialization |
| `conversion/clap.py`, `conversion/clamp3.py`, `conversion/mert.py`, `conversion/mert2.py`, `conversion/muq.py` | Model-specific config validation, tensor mapping and conversion |

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

### Q8_0 and Q4_K_M quantization (all architectures)

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
# CLaMP3 also accepts --audio; its bundled MERT tensors remain F32/F16.
```

All six native architectures (`clap`, `clamp3`, `mert`, `mert2`, `muq`,
`muq_mulan`) accept both presets. For CLAP/CLaMP3, the existing policy remains
unchanged. Their Q8_0 is **mixed precision**: tensors with ggml rank greater than one and a
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

For MERT, MERT2, MuQ and MuQ-MuLan, only recognized **2D dense/embedding**
weights are eligible: contiguous input width must be divisible by 32 (Q8) or
256 (K). Convolutions (including pointwise Conv1d), frontend/mel buffers,
normalization, GRN, vectors and incompatible widths retain their original bytes.
Quantized convolution/frontend/normalization layouts are rejected, and every
loaded tensor must be covered by the architecture's shape schema.

Their K policy is `q4_k_m-v2-encoder-dense-v-ffn-down-projection-q6_k`:

| Architecture | Eligible roles protected with Q6_K (all layers) |
| --- | --- |
| MERT | attention V, FFN output, feature projection |
| MERT2 | attention V, both Conformer FFN outputs, ConvNeXt dense output |
| MuQ | attention V, both FFN outputs, subsampler dense projection |
| MuQ-MuLan | all MuQ roles, XLM-R V/FFN output, adapter fused KV/GEGLU output, audio/text latent and audio adapter projections |

Other eligible dense/embedding matrices use Q4_K. The fused MuLan KV matrix
protects both K and V because they share one tensor. There is no Q8 fallback.
CLaMP3's mandatory bundled MERT is deliberately **not** quantized; its original
F32/F16 bytes and the pre-existing CLAP/CLaMP3 policy are preserved.

The loader keeps Q8/K weights mapped (Windows uses its existing file buffer).
CLAP/CLaMP3 linear layers use native ggml quantized matrix multiplication.
MERT/MERT2/MuQ/MuLan retain **F32 activations**: each dense operation decodes only
its current quantized operand to F32 scratch, runs native ggml F32 matmul, then
releases that scratch. This avoids hidden Q8 activation packing and is not a
whole-model expansion or a promise of faster inference. Temporary operands
never enter the persistent GPU weight cache. Scalar reads
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

`all_architecture_quantization` adds default, ML-package-free canonical byte,
role-policy, nonzero Q4/Q6, F16 byte-preservation, strict coverage and unsupported
convolution tests for all four additional encoders. With all four encoder oracle
options enabled, `encoder_quantized_parity` builds 256-wide complete networks,
tests both presets from F32/F16 sources, checks against independent
storage-rounded F32-activation references and exercises canonical F32 decoded
model roundtrips through the real CLI. It saves numerical metrics in the build
tree. These synthetic checks do **not** measure published-model quantization loss.
An all-equal, zero-statistics tiny MERT2 frontend has a known silence sensitivity:
the unquantized F32 reference discrepancy is approximately 0.0121 max absolute.
The suite retains that official-reference failure as an explicit diagnostic,
not accepted parity, while requiring strict quantized/decoded-native roundtrip
agreement for silence. Broadband official-reference checks remain strict.

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
./build/clap-model --model clamp3-f16.gguf --audio music.flac
```

Every CLaMP3 conversion **automatically bundles** `m-a-p/MERT-v1-95M`.
The converter resolves `config.json`, `preprocessor_config.json` and
`pytorch_model.bin` at the default immutable revision
`12af15fef9d0ac838c3f475bfbbf26d2060dd4f5` through Hugging Face's standard
cache, downloading missing files when online. `--mert-revision FULL_COMMIT_SHA`
can select another explicitly pinned revision of that same repository.
Use `HF_HOME`/`HF_HUB_CACHE` to locate an existing cache and `HF_HUB_OFFLINE=1`
to require cached inputs (a cache miss fails; bundling is never skipped).
Conversion does not execute remote model code. Both encoders use `--dtype`.

A single GGUF now handles both text and raw waveform inference, with native
mono decoding/resampling to 24 kHz and MERT's whole-recording normalization,
five-second windows and all-layer pooling described below. No external MERT
model or precomputed feature input is accepted. The public feature-input API
and its CLI flag have been removed; old feature-only GGUFs must be reconverted.
`Model::audio(pcm, rate)` is the C++ waveform entry point.

The bundle's string metadata contract is:

- `general.architecture = clamp3`; the existing CLaMP3 config/tokenizer remain unchanged.
- `clamp3.audio_frontend = mert-v1` selects the fixed whole-recording/5-second/all-layer extraction contract.
- `clamp3.mert.config` contains the validated original MERT JSON config.
- `clamp3.mert.source_revision` contains the full lowercase dependency commit SHA.
- `mert.*` tensors retain the standalone MERT names under one prefix. They share
  the same GGUF mapping and are consumed by the existing native `MertModel`, not
  a duplicate encoder. Missing/unsupported metadata, incompatible dimensions,
  missing/misshaped tensors and unused tensor names fail during model loading.

Quantization applies only to eligible CLaMP3 encoder/projection tensors; the
bundled MERT frontend stays in its original F32/F16 storage format.

The runtime adds
the official leading/trailing zero vectors, uses overlapping final 128-frame
chunks, masked mean pooling, remainder-weighted chunk aggregation, and a linear
projection. Unlike CLAP output, CLaMP3 returns a **768-dimensional**, non-L2-normalized
projection. Text uses the XLM-R Unigram vocabulary and its serialized Precompiled
normalizer, with the tokenizer embedded in GGUF and no runtime network access.
Text longer than a chunk is processed in full rather than truncated at 128/512 tokens.

The converter safely memory-maps the checkpoint, reads a nested `model` state
dictionary, omits the unsupported symbolic encoder, and supports `--dtype f32`
and `--dtype f16`. Symbolic ABC/MTF/MusicXML/image input is not supported.

### MERT-v1-95M and native CLaMP3 audio

The MERT-v1 encoder implements the official 95M, post-layer-normalized HuBERT
architecture at Hugging Face revision
[`12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5).
Download that snapshot's `config.json`, `preprocessor_config.json` and
`pytorch_model.bin` into `mert-checkpoint/`. The published weight SHA-256 is
`a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd`.
Conversion only parses metadata and safely memory-maps weights; it does not
import checkpoint Python code or use `trust_remote_code`.
The model card declares **CC-BY-NC-4.0** for the published MERT weights; this
project's Apache-2.0 license does not relicense those weights.

```sh
python convert_hf_to_gguf.py mert-checkpoint/ mert-v1-95m-f32.gguf \
  --dtype f32 --mert-revision 12af15fef9d0ac838c3f475bfbbf26d2060dd4f5
./build/clap-model --model mert-v1-95m-f32.gguf --audio music.wav --frames
./build/clap-model --model mert-v1-95m-f32.gguf --audio music.wav --clamp3-features
./build/clap-model --model clamp3-f16.gguf --audio music.wav
```

`--mert-revision` records the full source SHA in GGUF. Standalone local MERT
conversion treats it as caller-supplied provenance, not a content-hash check.
CLaMP3 conversion resolves that immutable revision from the fixed official repository;
modified local cache contents are still the caller's responsibility.
`--dtype f16` is also supported: dense/convolution weights use half storage,
with F32 activations/accumulation and vector parameters. The positional
convolution's legacy or parametrized weight normalization is materialized before
storage rounding. The checkpoint's training-only masking vector is omitted.
No Python, Torch, FFmpeg process, runtime download or external resampling library
is required for inference.

Two different outputs are intentionally exposed:

- `--frames` / `Model::audio_frames(pcm, rate)` normalize the complete supplied
  waveform and return the **last transformer layer**, in time-major order.
  The published model has 768 channels and frame count
  `floor((samples - 400) / 320) + 1` for at least 400 samples at 24 kHz.
  Without either output flag, `Model::audio()` / the CLI mean-pool these final
  frames into a non-normalized 768-vector. This is **not** a CLaMP3 input vector.
- `--clamp3-features` / `Model::clamp3_features(pcm, rate)` implement the official
  CLaMP3 `preprocessing/audio/extract_mert.py --mean_features` contract at
  [`sanderwood/clamp3@9016d2b0c8d12d1aa79c2e0ab201e6822bdc83a8`](https://github.com/sanderwood/clamp3/tree/9016d2b0c8d12d1aa79c2e0ab201e6822bdc83a8/preprocessing/audio):
  mix to mono and resample to 24 kHz, normalize the **whole recording** once as
  `(x - mean) / sqrt(population_variance + 1e-7)`, split non-overlapping 120,000-sample
  (five-second) windows, and discard only a final window shorter than 24,000
  samples (one second). For each retained window, mean over time and then all
  **13 hidden states**, including the projected/position-encoded encoder input.
  Return one 768-vector per window. At least one second of audio is required.

Both sequence flags print `{ "frames": N, "dimension": D, "values": [...] }`;
`values` is flat, time-major, and matches `clap::AudioFrames`. These standalone
MERT inspection APIs remain available; CLaMP3 feature encoding itself is private.
For C++ end-to-end inference:

```cpp
clap::Model clamp3("clamp3-f16.gguf");
auto pcm = clap::decode_audio("music.flac");
auto embedding = clamp3.audio(pcm.samples, pcm.sample_rate);
```

CLaMP3 waveform inference includes the leading/trailing zero vectors, chunking
and raw projection internally. CLaMP3 rejects `--frames`, `--clamp3-features`
and `--crop-offset` (including an explicitly supplied zero). Standalone MERT
sequence flags also exclude `--crop-offset`, and cannot be combined together.
Native decoding/resampling uses miniaudio as documented below, **not** the
bit-identical librosa/torchaudio path of the Python extraction script. Numerical
oracles compare identical 24 kHz PCM, and separately test 48 kHz input against a
native-resampler control.

The optional offline oracle uses locally reviewed `modeling_MERT.py` and
`configuration_MERT.py` from the pinned snapshot (no network access):

```sh
cmake -S . -B build -DMERT_MODEL_PARITY=ON -DCLAMP3_MODEL_PARITY=ON \
  -DMERT_REFERENCE_SOURCE=/path/to/reviewed/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5
cmake --build build --parallel
ctest --test-dir build --output-on-failure
# Actual 95M checkpoint, not the random tiny model:
python tests/models/mert_parity.py build build/published-mert-v1 \
  /path/to/reviewed/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5 --checkpoint mert-checkpoint/
# Actual published MERT + CLaMP3 SAAS weights; use a bundled F32 CLaMP3 GGUF:
python tests/models/mert_checkpoint_pipeline.py build mert-checkpoint/ clamp3-checkpoint/ \
  /path/to/reviewed/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5 build/published-pipeline \
  --mert-gguf mert-v1-95m-f32.gguf --clamp3-gguf clamp3-f32.gguf
```

The tiny F32/F16 suites cover waveform normalization, convolutions, attention,
final frames, all-layer means, five-second boundaries/tail discard, public APIs,
CLI validation, native/resampled pipelines and Transformers BERT projection.
The CLaMP3 fixtures populate an isolated offline Hugging Face cache with tiny
synthetic MERT weights; no model download is needed for these tests. Private
test-only feature-encoder access preserves independent tensor-level oracles
without exposing feature input in the library or production executable.
Always-on tests reject malformed model/converter metadata without Torch.
A local published-95M check (broadband, tone, silence and resampled audio) had
worst frame errors below **4.3e-5** and all-layer feature errors below **5.1e-6**, for both F32 and
storage-rounded F16 oracles. Published MERT (F32/F16) chained to published CLaMP3
SAAS F32 matched the independent Transformers pipeline within **3.4e-6** on
one-second broadband, six-second/two-window audio and resampled input. These
synthetic-waveform checks establish inference parity, not retrieval quality.

Scope: MERT-v1-95M only; the 330M variant, stable-layer-norm/DeepNorm, CQT frontend,
batching, arbitrary layer selection and full GPU execution
are not supported. Full-recording `--frames`/pooled inference has quadratic
attention time cost; prefer the bounded five-second `--clamp3-features` path for
CLaMP3/long recordings. This does not change MERT-v2-FullSong behavior.

### MuQ-large-msd-iter

`OpenMuQ/MuQ-large-msd-iter` is an **audio-only** representation model, not
MuQ-MuLan. Its published weights are **CC-BY-NC-4.0 (non-commercial)**; the
upstream implementation is MIT. Adding a native loader does not change the
weight license. No checkpoint is bundled with this project.

Convert a local safetensors snapshot pinned to Hugging Face revision
`0562a57814f6f8bbd9fdea0a25921a2fce1a841a`:

```sh
python -m pip install -r requirements.txt
python convert_hf_to_gguf.py checkpoint/ muq-f32.gguf \
  --muq-revision 0562a57814f6f8bbd9fdea0a25921a2fce1a841a
./build/clap-model --model muq-f32.gguf --audio music.flac
./build/clap-model --model muq-f32.gguf --audio music.flac --frames
# F16 saves disk space; MuQ expands it to F32 compute weights on load:
python convert_hf_to_gguf.py checkpoint/ muq-f16.gguf --dtype f16 \
  --muq-revision 0562a57814f6f8bbd9fdea0a25921a2fce1a841a
```

The frontend mixes to mono/resamples to **24 kHz**, then computes the official
periodic-Hann 2048-point, centered-reflect power spectrogram, HTK 128-bin mel
projection, power dB without top-db clipping, last-frame removal, and fixed
published mean/std normalization. There is no waveform amplitude standardization.
The encoder has two residual Conv2d subsampling stages with evaluation BatchNorm,
followed by 12 width-1024 Wav2Vec2-Conformer layers. In this architecture RoPE is
applied **before Q/K projections**, not after as in MERT2; feed-forward and
convolution branches use SiLU. Encoder final LayerNorm is included.

`Model::audio_frames()` and `--frames` return the final unnormalized sequence;
for `N` native-rate samples its frame count is `ceil(floor(N / 240) / 4)` (about
25 Hz), not MERT2's floor rule. At least 1,025 samples are required.
`Model::audio()` and the default CLI output mean-pool all returned frames into
one 1,024-vector, without L2 normalization. Pooling is a convenience over MuQ
features, **not** a MuQ-MuLan text/audio projection. Text, precomputed features,
batching, training/RVQ codes, and intermediate-layer selection are unsupported.
The entire supplied waveform is processed, without an implicit crop or chunk
aggregation; bidirectional attention has quadratic compute and linear score
workspace. Conv2d im2col scratch is tiled, but feature maps still grow with input
length. Long inputs can require substantial RAM and CPU time.

F32/F16 storage is supported; frontend buffers and vector parameters remain
F32. MuQ F16 weights are expanded to owned F32 buffers on load so GGML does not
also round activations to F16 during matrix multiplication. F16 therefore saves
disk space, **not runtime weight RAM**; it does not retain F32's lazy mmap weight
behavior and may use additional RAM for the original file buffer on Windows.
Conversion streams one safetensors tensor at a time, excluding training-only
RVQ/classifier/CLS tensors and positional-convolution parameters unused by the
official inference forward. Q8_0 and mixed Q4_K_M dense weights are supported
with operation-local F32 scratch (see quantization above); noneligible F16
convolutions retain the existing promotion behavior. Dense weights use the shared
CPU linear backend in F32-activation mode; convolutions, frontend and attention remain
CPU operations. This is not end-to-end GPU inference.

The optional oracle uses SHA-verified `conv.py` and `features.py` from official
MuQ source revision `28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a`, wired to the
Transformers Wav2Vec2-Conformer exactly as the official inference path. It does
not instantiate unused training/RVQ modules. The tested oracle environment is
torch/torchaudio 2.6.0 and Transformers 4.53.2, with `einops` installed:

```sh
cmake -S . -B build -DMUQ_MODEL_PARITY=ON \
  -DMUQ_REFERENCE_SOURCE=/path/to/pinned/MuQ/source-components
cmake --build build --parallel
ctest --test-dir build -R muq --output-on-failure
python tests/models/muq_checkpoint_parity.py build checkpoint/ \
  /path/to/pinned/MuQ/source-components build/published-muq
# Optional identically storage-rounded F16 reference:
python tests/models/muq_checkpoint_parity.py build checkpoint/ \
  /path/to/pinned/MuQ/source-components build/published-muq-f16 --dtype f16
```

Tiny-model tests cover learned BatchNorm statistics, silence, odd frame counts,
a real-rate 31-second input, F32/F16 storage, resampling, API and CLI. Those random
weights are not published-model evidence. The separate published oracle checks
weight SHA-256 `273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6`
and compares a 5,041-sample waveform (six final frames), not long-music quality.
The default published fixture includes seeded low-level broadband energy so that
its strict absolute-error test measures the architecture rather than differences
at the power-dB noise floor. Native FFT intermediates use double precision to
avoid accumulating single-precision twiddle error. This is not bit-exact with
PyTorch's float32 FFT: a published-weight pure-tone diagnostic measured a frame
max error of **0.013963**, failing the same `5e-4` F32 limit. Reproduce that known
near-noise-floor limitation with `--tone-diagnostic`; its strict tolerance is
not widened. A passing broadband test does not claim that error bound for every
waveform or establish retrieval quality.
Both oracle commands are offline and write generated audio/fixtures only to their
output directories. Ordinary CTest also covers config/license metadata and
quantized-storage rejection without requiring ML packages.

### MuQ-MuLan-large (native joint audio/text embeddings)

`OpenMuQ/MuQ-MuLan-large` has native, single-item text and raw-waveform
inference through `clap::Model` and `clap-model`. Published embeddings have 512
components. No Python, model source execution, or downloads occur at inference.
The model revision is `2e01c796b71dca71b45251384c04cd7b237c9020`; the reviewed
[official source](https://github.com/tencent-ailab/MuQ/tree/28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a)
is pinned to `28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a` (MIT). The
[weights](https://huggingface.co/OpenMuQ/MuQ-MuLan-large/tree/2e01c796b71dca71b45251384c04cd7b237c9020)
are **CC-BY-NC-4.0**, including the resulting GGUF. Attribution: Haina Zhu et al.,
*MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector
Quantization*. This support does not change the repository's own license.

Conversion requires these **local** files:

- MuLan `config.json` and the complete `pytorch_model.bin` in the source directory.
- The MuQ-large-msd-iter dependency's `config.json`, supplied with `--muq-config`
  (default: `muq-config.json` in the source directory).
- XLM-R-base `config.json`, supplied with `--text-config` (default:
  `text-config.json` in the source directory), and its exact `tokenizer.json`,
  supplied with `--tokenizer` (default: beside the checkpoint).

The published checkpoint is 2,653,954,401 bytes, SHA256
`d42ae3f7cb9b66759ee0089ddc70e2f28b130c2d8ba621457358272d32dd0444`.
Use the XLM-R-base vocabulary with 250002 entries, hidden size 768, 12 backbone
layers and 514 position entries. The MuQ backbone has hidden size 1024 and 12
layers. Conversion validates the dependency dimensions against every inference
tensor, rejects unknown tensors, and ignores only known unused/training
namespaces. It uses `torch.load(weights_only=True, mmap=True)` and local JSON;
it never instantiates a pretrained model or executes downloaded source. Missing
dependency configs cause an error, rather than a hidden download.

```bash
python convert_hf_to_gguf.py /path/to/muq-mulan artifacts/muq-mulan-f16.gguf \
  --dtype f16 \
  --muq-mulan-revision 2e01c796b71dca71b45251384c04cd7b237c9020 \
  --muq-config /path/to/muq/config.json \
  --text-config /path/to/xlm-roberta-base/config.json \
  --tokenizer /path/to/xlm-roberta-base/tokenizer.json

artifacts/build/clap-model --model artifacts/muq-mulan-f16.gguf \
  --text 'classical genres, hopeful mood, piano.' --threads 2
artifacts/build/clap-model --model artifacts/muq-mulan-f16.gguf \
  --text '一首适合海边风景的小提琴曲，节奏欢快' --threads 2
artifacts/build/clap-model --model artifacts/muq-mulan-f16.gguf \
  --audio music.flac --threads 2
```

```cpp
clap::Model model("artifacts/muq-mulan-f16.gguf", 2);
auto text = model.text("hopeful piano music");
auto audio = clap::decode_audio("music.flac");
auto music = model.audio(audio.samples, audio.sample_rate);
// dot(music, text) is the official similarity operation.
```

Audio is mixed to mono and resampled to 24 kHz using the existing native helper.
There is **no waveform amplitude normalization**. The helper's causal filter and
linear resampling are not bit-equivalent to torchaudio/librosa; compare at a
common, pre-resampled 24 kHz waveform for a backbone/projection parity check.
The model splits successive ten-second windows. A final tail gets **one slice**
from the recording prefix, of at most the missing length. A recording shorter
than five seconds therefore produces a clip with `2*N` samples, rather than a
repeated or zero-padded ten-second clip. Exactly ten seconds produces one clip.
Each clip uses the final normalized MuQ layer, the 1024→768 affine audio
projection, mean pooling, the learned 768→512 affine projection, and L2
normalization. Clip vectors are averaged **without duration weighting or a
second normalization**; a multi-clip result can have norm below one. At least
1025 samples are needed **after** clipping: 513 original samples are accepted
at 24 kHz, while 512 are rejected. Empty/nonfinite input is rejected.

Text uses the existing exact XLM-R Unigram tokenizer and XLM-R-base backbone,
followed by eight custom prenorm transformer blocks with cosine Q/K attention,
learned Q/K scales, score multiplier 8 and GEGLU. The custom transformer has no
additional padding mask for the single-item API, and no final adapter norm.
Mean pooling includes special tokens, then the learned 768→512 affine
projection and L2 normalization apply. Empty text is valid. Over-context text
is rejected, rather than truncated or chunked; the published XLM-R table allows
512 tokenizer IDs including specials. `Model::tokenize(text, max_length)` keeps
its existing explicit truncation API; `Model::text(string)` tokenizes the full
input and checks the context. Batch inference and other backbone/use-layer
variants are unsupported.

Both F32 and F16 GGUF are supported. Frontend buffers and 1D parameters remain
F32. To preserve storage-rounded F32-activation semantics, F16 MuQ backbone
weights expand into owned F32 compute buffers; text embeddings stay mapped,
and F16 dense matrices expand only for the current operation. This saves disk
and avoids expanding the entire joint model, but still requires roughly the
MuQ backbone's F32 size in RAM, plus mapped working pages, activations, tokenizer
and scratch. F16 is not an assurance that this large model fits in a small RAM
budget without paging. MuLan currently uses the validated CPU dense path.
Q8_0 and mixed Q4_K_M dense/embedding storage are supported (see quantization
above), without expanding the entire quantized model. Frame output, CLaMP3 features, precomputed
features and crop offsets are unsupported for this joint model.

The optional tiny oracle exercises official hash-checked components with
nontrivial random learned weights; it does not establish published retrieval
quality. The ordinary CTest contracts require only Python's standard library.

```bash
cmake -S . -B artifacts/build \
  -DMUQ_MULAN_MODEL_PARITY=ON \
  -DMUQ_MULAN_REFERENCE_SOURCE=/path/to/pinned/MuLan/source \
  -DMUQ_REFERENCE_SOURCE=/path/to/pinned/MuQ/source-components
cmake --build artifacts/build -j2
ctest --test-dir artifacts/build -R muq_mulan --output-on-failure

# Separate low-RAM published-checkpoint oracle. Each Torch branch runs in a
# child process and exits before the native CLI starts; arrays/metrics are saved.
python tests/models/muq_mulan_checkpoint_parity.py \
  --checkpoint artifacts/checkpoint --gguf artifacts/muq-mulan-f16.gguf \
  --dtype f16 --cli artifacts/build/clap-model \
  --source artifacts/oracle --muq-source /path/to/pinned/MuQ/source-components \
  --out artifacts/published-muq-mulan-f16 --audio music.wav
```

For the published oracle, the checkpoint directory contains dependency files
named `muq_config.json`, `text_config.json`, and `tokenizer.json`. The script
verifies the complete checkpoint's size/hash, constructs only one branch at a
time on the meta device, assigns mmap weights, and checks F16 against
storage-rounded F32 activations. It also reports the difference between native
and Python resampling separately. Actual-weight parity requires running this
oracle with the pinned checkpoint; synthetic fixtures alone do not supply that
evidence.

Published-checkpoint checks were run with the verified source/weight revisions
above, five single-item text probes (English, Chinese, empty, combining Unicode
and a special token), and identical 24 kHz PCM for a seeded 264137-sample
synthetic waveform spanning two clips, including prefix-wrapped tail handling:

| GGUF storage | Worst text absolute error | Audio absolute error |
| --- | ---: | ---: |
| F32 | 3.50177e-7 | 2.45869e-7 |
| F16 (storage-rounded F32-activation reference) | 3.58094e-7 | 2.11410e-7 |

All outputs have 512 components. The two-clip native audio norms are
`0.979717080` (F32) and `0.979694188` (F16), rather than one, consistent with the
official unnormalized clip mean. These are inference-parity probes, not a
universal waveform error bound or an assessment of retrieval quality. A separate
5041-sample two-tone diagnostic without added broadband energy measured audio
errors of `1.84137e-4` (F32) and `1.89982e-4` (F16), both below the unchanged
`5e-4` diagnostic limit but substantially above the broadband probe errors.
The MuQ frontend FFT/noise-floor and resampler limitations described above
still apply.

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
parameters and all official float32 frontend tensors remain F32. MERT2 CPU
matmuls expand only the current F16 operand to F32 scratch, preserving F32
activation semantics. The FFT uses double intermediates and rounds its output
to F32 to avoid repeated-twiddle drift in the spectral noise floor; a direct-DFT
regression covers 2048-point complex and log-power errors. This incorporates the
previously separate MERT2 FFT/activation-precision correction into the runtime;
it does not retroactively validate old binaries or promise bitwise agreement
with every Torch FFT backend. The converter
streams safetensors one at a time rather than materializing the 632M-parameter
checkpoint in RAM. Q8_0 and mixed Q4_K_M dense storage are supported (see above).
Batching, intermediate hidden-state
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

For raw music files, use the native `--mert-model` pipeline above. Alternatively,
run `preprocessing/audio/extract_mert.py` in the official `sanderwood/clamp3`
checkout with `--model_path m-a-p/MERT-v1-95M` and `--mean_features`, then pass
its `.npy` to `--audio-features`. Existing dummy interfaces below are unchanged.

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
