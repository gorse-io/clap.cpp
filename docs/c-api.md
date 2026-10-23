# Public C API

`include/clap.h` is an ISO C99 header and the only installed clap API header.
It follows the opaque-handle, `extern "C"`, export-macro and plain parameter
struct conventions used by llama.cpp and stable-diffusion.cpp. The implementation
still uses C++17/ggml; STL objects, exceptions and allocator ownership never cross
the public interface. The former `clap::Model` API is now private in
`src/clap-internal.h`; this is an intentional source-breaking API change, not a
C++ compatibility wrapper. The CPython extension itself is not included.

## Build and link

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build --parallel
cmake --install build --prefix "$PWD/stage"
```

An installed CMake consumer can use:

```cmake
cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES C CXX)
find_package(clap CONFIG REQUIRED)
add_executable(example main.c)
set_target_properties(example PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO LINKER_LANGUAGE CXX)
target_link_libraries(example PRIVATE clap::clap)
```

Configure with `-DCMAKE_PREFIX_PATH=/path/to/stage`. `main.c` is compiled as C;
the C++ linker supplies the static implementation's C++ runtime. No C++ compile
features or implementation include directories are usage requirements of `clap`.
Shared-library C consumers can instead use their C linker directly. Outside
CMake, define `CLAP_SHARED` when consuming a Windows DLL, but not a static library;
`CLAP_BUILD` is reserved for building the DLL. CMake supplies these definitions.
Static installations include ggml and the bundled ICU archives required at link
time. Shared installations include the ggml shared libraries; installed Unix
binaries use relative runtime library paths. With `CLAP_USE_SYSTEM_ICU=ON`, ICU is
an external dependency instead.

## Text inference

```c
#include <clap.h>
#include <stdio.h>

int main(int argc, char **argv) {
    struct clap_model *model = NULL;
    struct clap_float_buffer embedding = {0};
    struct clap_model_params params = clap_model_default_params();
    int result = 1;
    size_t i;
    if (argc != 2) return 2;
    params.n_threads = 2;
    if (clap_model_load(argv[1], params, &model) != CLAP_STATUS_SUCCESS)
        goto error;
    if (clap_model_text(model, "A jazz quartet.", &embedding) != CLAP_STATUS_SUCCESS)
        goto error;
    for (i = 0; i < embedding.size; ++i) printf("%g\n", embedding.data[i]);
    result = 0;
    goto cleanup;
error:
    fprintf(stderr, "%s\n", clap_last_error());
cleanup:
    clap_float_buffer_free(&embedding);
    clap_model_free(model);
    return result;
}
```

Every model load selects its architecture from GGUF metadata, as before.
`clap_model_tokenize` plus `clap_model_text_tokens` also supports explicit token
inference. Use `max_length=512` for the former tokenizer default; truncation is
model/tokenizer-specific. String inference uses the model's context policy.

## Audio and frame outputs

- `clap_audio_decode` decodes WAV/MP3/FLAC into mono PCM without resampling.
- `clap_audio_resample` exposes the existing causal linear+LPF resampler.
- `clap_log_mel` returns the existing flattened `[1001,64]` Slaney features;
  `clap_fft` exposes the in-place double-precision forward FFT with explicit
  real/imaginary fields, not C99 complex or `std::complex` layout assumptions.
- `clap_model_audio` accepts borrowed mono PCM and automatically resamples.
  Initialize its parameters with `clap_audio_default_params()` (48 kHz, crop 0),
  then set `sample_rate` to the actual input rate. Crop offsets are native-rate
  samples and only supported for CLAP. MERT/MuQ use all final-layer frames;
  MuLan retains its clip pooling. No normalization policy changes are made:
  CLAP unit-normalizes, CLaMP3 does not, MuLan averages normalized clips.
- `clap_model_audio_frames` exposes MERT-v1/v2 and MuQ final-layer sequences.
  Values are time-major with `n_values == frames * dimension`.
- `clap_model_clamp3_features` exposes standalone MERT-v1 all-layer/time mean
  features in five-second windows; it is not the final-layer frame API.

Unsupported model capabilities return an error, rather than fabricated results.
`examples/embedding/embedding.cpp` uses this C API for the native CLI, including
all audio/frame/features paths.

## Ownership, errors and threads

The header is the normative contract:

- Inputs are borrowed for a call. Output structs must start as `{0}`; each call
  resets its output before validation. On failure it stays empty. **Free an old
  result before reusing an output**; resetting does not free the old allocation.
  Inputs and outputs must not alias.
- Free results using the matching `clap_float_buffer_free`,
  `clap_token_buffer_free`, `clap_audio_free` or `clap_audio_frames_free`.
  These accept NULL and reset structs, so freeing a reset struct again is safe.
  Do not copy owning structs and free both copies or use Python/libc `free`.
  Free a model exactly once with `clap_model_free`; NULL is accepted.
- Every operation returns `CLAP_STATUS_SUCCESS`, `CLAP_STATUS_INVALID_ARGUMENT`,
  `CLAP_STATUS_OUT_OF_MEMORY` or `CLAP_STATUS_ERROR`. No C++ exceptions escape.
  Failure reporting uses a fixed 1024-byte thread-local buffer and cannot itself
  allocate, even while handling allocation failure. Messages may be truncated.
- `clap_last_error()` borrows that buffer until the next status-returning call on
  the same thread (or thread exit). Success clears it. Defaults, frees and error reads leave it
  unchanged. Copy a diagnostic before doing another operation on that thread.
- Calls on one model are serialized by a per-handle mutex, including tokenization
  and metadata getters. Different handles and standalone helpers can run in
  parallel. Never overlap model destruction with a call; callers own lifetime
  synchronization and must not concurrently mutate shared input/output storage.

For CPython/ctypes, declare `argtypes` and `restype` (including by-value parameter
structs); use `c_size_t` for lengths and 32-bit integers for sample rates and
IDs. `ctypes.CDLL` releases the GIL, so the native mutex matters. Copy result data
before calling the matching library free. `tests/c_api.py` is an executable
stdlib-only binding example, not an installed Python package.

## Regression checks

```sh
# Static and shared default suites; ctypes inference is enabled in shared builds.
cmake -S . -B build-static -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build-static --parallel
ctest --test-dir build-static --output-on-failure
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build-shared --parallel
ctest --test-dir build-shared --output-on-failure

# Fresh production build: no Python/FFmpeg discovery or test fixture generation.
cmake -S . -B build-production -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF
cmake --build build-production --parallel

# Staged external C99 consumer (repeat for static and shared builds).
cmake --install build-static --prefix "$PWD/build-static/stage"
cmake -S tests/installed -B build-static/consumer \
    -DCMAKE_PREFIX_PATH="$PWD/build-static/stage"
cmake --build build-static/consumer
python3 tests/c_api_fixture.py build-static/consumer/model.gguf
build-static/consumer/clap-installed-c build-static/consumer/model.gguf
```

`clap_c_api` compiles a real `.c` consumer with C99, extensions disabled and
strict diagnostics. It checks defaults, null/invalid inputs, exact helper
results, ownership resets and optional real-model inference. `clap_c_api_oom`
forces C++ allocation failure to verify the nonallocating boundary (static and
Linux shared builds). `clap_c_api_ctypes` exercises actual exported symbols,
Unicode tokenization, token/string equivalence, waveform-capable CLAP inference
at native/resampled rates, MERT frames/pooling/features, thread-local errors and
concurrent calls on one handle. Its valid GGUF and WAV fixtures are generated
with stdlib Python; no model downloads, Torch or committed binary fixtures are
needed. Existing numerical/model regression tests continue to use the private
implementation API.
