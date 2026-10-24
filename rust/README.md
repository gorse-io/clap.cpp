# Rust bindings

`clap-cpp` provides safe, owning CPU bindings to the current C++ `clap::Model`
API. It does **not** depend on the proposed public C API or a system installation
of clap.cpp. Rust has no third-party crate dependencies.

## Build and use

Requirements: Rust stable, CMake >= 3.20, a C/C++17 compiler, and a complete
repository checkout with these pinned submodules initialized:

```sh
git submodule update --init --recursive ggml thirdparty/json thirdparty/icu thirdparty/miniaudio
cargo test --manifest-path rust/Cargo.toml -j 4
cargo run --manifest-path rust/Cargo.toml --example embed -- model.gguf "a piano" audio.wav
```

Python 3 (stdlib only) is needed **only for integration tests**, which generate
tiny valid GGUF models and WAV data without downloading weights. Windows uses
the MSVC Rust toolchain and Visual Studio C++ build tools; macOS uses the Xcode
command-line tools. Native cross-compilation is not currently supported. Choose
moderate Cargo job counts to bound CMake compilation parallelism.

In another local project:

```toml
[dependencies]
clap-cpp = { path = "/path/to/clap.cpp/rust" }
```

```rust,no_run
use clap_cpp::{Model, decode_audio};
# fn main() -> clap_cpp::Result<()> {
let mut model = Model::new("model.gguf", 4)?;
let tokens = model.tokenize("a piano", 512)?;
let embedding = model.text_tokens(&tokens)?;
let audio = decode_audio("piano.wav")?;
let embedding = model.audio(&audio.samples, audio.sample_rate, 0)?;
# Ok(())
# }
```

The independent CMake entry point builds only core, CPU ggml, bundled ICU/json,
and the private bridge. Libraries are static and PIC; GPU backends, host-native
ISA tuning, BLAS/Accelerate, OpenMP, CLI tools and native tests are disabled.
System C/C++ runtimes are still required; no clap/ggml/ICU shared libraries or
ICU data files are needed. CMake's usual `CC`, `CXX`, `CFLAGS`, `CXXFLAGS`,
`CMAKE_GENERATOR` and `CMAKE_TOOLCHAIN_FILE` settings can configure native builds.

## API and safety

- `Model::new(path, threads)` loads one GGUF; `Drop` releases it. Zero threads
  selects the native default. Negative thread counts are rejected.
- Models are deliberately **neither `Send` nor `Sync`**. Inference takes
  `&mut self`; do not bypass this with unsafe code. Backend same-instance
  concurrency is not guaranteed.
- `tokenize`, `text`, `text_tokens`, `audio`, `audio_frames`,
  `clamp3_features`, `audio_sample_rate`, `projection_dimension` mirror native
  operations. Unsupported architecture/operation combinations return errors.
  `tokenize` truncates to a maximum, but does **not** pad. String inference uses
  the native architecture's context policy. Empty text can be valid; empty
  token slices are not.
- Frame results are time-major `AudioFrames { values, frames, dimension }`.
  `values.len() == frames * dimension`. MERT final frames and all-layer CLaMP3
  preprocessing features are different operations; CLAP does not expose either.
- `decode_audio` supports WAV/MP3/FLAC and averages channels, preserving source
  rate. `resample_audio` uses the native causal linear/anti-alias resampler;
  its output is not zero-phase sinc. `log_mel` accepts 48 kHz mono PCM and
  returns flattened `[1001, 64]` features with native repeat-padding/cropping.
  `audio` performs model-specific resampling; crop offsets retain the native
  model's meaning (CLAP: samples after resampling to 48 kHz). Other architectures
  may reject nonzero crop offsets.
- Native exceptions are caught inside the private bridge. Every result has a
  paired C++ free; output vectors and diagnostics are copied into Rust-owned
  values before release, never borrowed from a destroyed native object. An
  allocation failure while creating a native result becomes an `Error`; native
  diagnostic text is bounded to 1023 bytes. Rust allocation failure retains
  Rust's standard process-abort behavior.
- Strings and paths must be UTF-8 and cannot contain NUL. Non-UTF-8 OS paths
  are rejected rather than lossy-converted. Input slices are borrowed only for
  synchronous calls. PCM must be nonempty and finite with positive sample rate;
  invalid tokens and architecture/context constraints are checked natively.

## Packaging limitation (intentional)

This is a **repository/path dependency**, not a standalone crates.io package.
`publish = false` prevents publishing an unusable source archive. Cargo packages
only files inside `rust/`; the parent C++ sources and Git submodules are therefore
absent from `cargo package` archives. A package archive cannot build standalone,
and a plain Cargo Git dependency does not reliably initialize the required
submodules. Use an explicitly cloned, initialized checkout and a path dependency.
Do not remove the publishing guard without first adding a reviewed native-source
vendoring/release process, all dependency licenses, and an isolated package-build
test. `cargo package --list --allow-dirty` can inspect the archive inventory;
that is not evidence of standalone package viability.

## Checks

```sh
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml
```

Tests exercise actual CLAP text/waveform and MERT inference, independent scalar
text values, dimensions, frame mean-pooling, decoding/resampling/mel values,
UTF-8 paths, owned errors, unsupported operations, invalid IDs and nonfinite PCM.
Compile-fail doctests enforce the !Send/!Sync contract. Large published models,
GPU inference and exhaustive numerical parity are outside this binding suite.
