# Rust bindings

`clap-cpp` provides safe, owning CPU bindings to the C++ `clap::CLAP` API.
The public surface is `CLAP`, its constructor, and `encode_text` / `encode_audio`,
using the Python binding's naming and file-encoding behavior. It does not depend
on a public C API or a system installation of clap.cpp. Rust has no third-party
crate dependencies.

## Build and use

Requirements: Rust stable, CMake >= 3.20, a C/C++17 compiler, and a complete
repository checkout with these pinned submodules initialized:

```sh
git submodule update --init --recursive ggml thirdparty/json thirdparty/icu thirdparty/miniaudio
cargo test --manifest-path rust/Cargo.toml -j 3
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
use clap_cpp::CLAP;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut model = CLAP::new("model.gguf", 4)?;
let text_embedding = model.encode_text("a piano")?;
let audio_embedding = model.encode_audio("piano.wav", 0)?;
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

- `CLAP::new(path, threads)` loads one GGUF; `Drop` releases it. Zero threads
  selects the native default. Negative thread counts are rejected.
- `encode_text(&str)` returns one owned `Vec<f32>` embedding, using the native
  architecture's context policy. Empty text can be valid.
- `encode_audio(path, crop_offset)` encodes a WAV, MP3 or FLAC file, decoding,
  averaging channels to mono and resampling to the model's rate internally.
  Use `0` for the default crop. CLAP crop offsets count samples after resampling
  to 48 kHz; other architectures may reject nonzero offsets.
- Unlike Python's overloads, this minimal Rust API accepts text strings and
  audio file paths only, not token IDs or PCM slices. There are no public
  tokenization, frame/features, metadata or standalone audio preprocessing APIs.
- Models are deliberately **neither `Send` nor `Sync`**. Inference takes
  `&mut self`; backend same-instance concurrency is not guaranteed.
- Native exceptions are caught inside the private bridge. Every result has a
  paired C++ free; embeddings and diagnostics are copied into Rust-owned values.
  Errors use `Box<dyn std::error::Error>`, not an additional public binding type.
  Native diagnostic text is bounded to 1023 bytes. Rust allocation failure
  retains Rust's standard process-abort behavior.
- Strings and paths must be UTF-8 and cannot contain NUL. Non-UTF-8 OS paths
  are rejected rather than lossy-converted. Inputs are borrowed only for
  synchronous calls. Unsupported architecture/operation combinations return errors.

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
cargo test --manifest-path rust/Cargo.toml --all-targets
cargo test --manifest-path rust/Cargo.toml --doc
```

Tests exercise actual CLAP text/file and MERT file inference, independent scalar
text values, Unicode paths, stereo downmixing at a non-model sample rate,
nonzero cropping on long audio, owned errors and unsupported operations.
Compile-fail doctests enforce the !Send/!Sync contract. Large published models,
GPU inference and exhaustive numerical parity are outside this binding suite.
