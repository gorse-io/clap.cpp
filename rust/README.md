# clap.cpp Rust bindings

[![crates.io](https://img.shields.io/crates/v/clap-cpp)](https://crates.io/crates/clap-cpp)
[![crates.io downloads](https://img.shields.io/crates/dm/clap-cpp)](https://crates.io/crates/clap-cpp)

## Install

```sh
cargo add clap-cpp
```

Requires CMake 3.20 or newer and a C++20 compiler (MSVC on Windows). The
published crate bundles clap.cpp, ggml, JSON, ICU Unicode/regex,
and miniaudio sources. Default builds enable Metal for macOS targets and remain
CPU-only on Linux and Windows. macOS builds require the Apple SDK and Metal
compiler. Default Linux and Windows builds need no GPU SDK. Builds do not download
dependencies or require Git, Python, shared ICU libraries, or the parent repository.
Cross-compiling is not currently supported. Python 3 is required only for
integration-test fixture generation and maintainer checks.

Rust declares the public C API directly with `unsafe extern "C"` and wraps model
handles and returned buffers in Rust RAII owners. There is no C++ Rust bridge or
separate `rust/native/CMakeLists.txt` build graph. Both checkout and packaged
builds configure the root `CMakeLists.txt` in core-only mode, without CLI or test
dependencies, and link the static clap/ggml/ICU archives. CMake supplies the
resolved transitive SDK link dependencies for enabled GPU features.

## GPU features

Enable an optional backend when adding the dependency (Metal is already enabled
by default on macOS):

```sh
cargo add clap-cpp --features cuda
# Or: --features vulkan, --features rocm, --features metal
```

| Feature | Native backend | Build requirements |
| --- | --- | --- |
| `cuda` | GGML CUDA | NVIDIA CUDA Toolkit and a supported C++ host compiler |
| `vulkan` | GGML Vulkan | Vulkan headers/loader, `glslc`, and SPIR-V headers |
| `rocm` | GGML HIP | AMD ROCm/HIP SDK with hipBLAS and rocBLAS |
| `metal` | GGML Metal (default on macOS; explicit feature still supported) | Apple SDK and Metal compiler on macOS |

`--no-default-features` does not disable the target-based macOS Metal default.
To force CPU inference at runtime, set `CLAP_DEVICE=cpu` before loading a model.

The CPU backend remains available with every feature. Features compile backends;
they do not guarantee that a compatible GPU is present or that every operation
runs on it. Backend selection follows clap.cpp's native behavior, without changes
to the Rust API. CUDA and ROCm cannot be enabled together; do not use
`--all-features` to test this crate. GPU SDKs and runtime drivers are external
requirements and are not bundled in the crate. Enabled backends may introduce
shared vendor runtime dependencies even though clap.cpp and ggml are linked
statically.

For checkout builds, use e.g. `cargo test
--features vulkan --all-targets -j 3`. SDK discovery uses CMake; configure
nonstandard installations with the SDK's usual environment variables (such as
`VULKAN_SDK` or `HIP_PATH`) or a `CMAKE_TOOLCHAIN_FILE`.

## Building from Git

Initialize the pinned submodules before using a path dependency or building:

```sh
git submodule update --init --recursive
cargo test --all-targets -j 3
cargo test --doc -j 3
```

`Cargo.toml` and `Cargo.lock` live at the repository root; Rust sources, build
script, tests, examples and this README remain under `rust/`. Use the repository
root for path dependencies. Cargo writes build output to root `target/` by default.
Checkout and registry builds use the same native source paths; no generated
source tree can override checkout edits.

## Packaging and publishing (maintainers)

From the repository root, with Python 3 and the pinned submodules initialized:

```sh
python3 -m unittest discover -s rust/tests -p test_package.py -v
cargo package --locked -j 3
```

The guard checks require clean submodules at the revisions pinned by the checkout
and check Cargo's actual file inventory. Cargo includes the initialized Git
submodules directly through the root manifest's include allowlist: the native
CMake graph, clap.cpp sources, selected dependency source trees (including optional
GPU backends), upstream licenses in their original locations, and
`python/tests/tiny_models.py` for standalone tests. There is no copying step and
no `vendor/` tree. Unrelated CLI dependencies, generated models, audio, build
output and Git metadata are omitted. Do not commit generated fixtures or archives.
For an intentional dirty local package use `--allow-dirty`; releases should use
the pinned clean source tree.

Do not skip Cargo's package verification: it builds the archived source rather
than the checkout. Also run integration tests against a standalone extraction
(no parent source tree), replacing the version below if needed:

```sh
mkdir -p target/standalone
tar -xzf target/package/clap-cpp-0.1.0.crate -C target/standalone
cargo test --manifest-path target/standalone/clap-cpp-0.1.0/Cargo.toml --all-targets --locked -j 3
cargo test --manifest-path target/standalone/clap-cpp-0.1.0/Cargo.toml --doc --locked -j 3
```

After review and verification, publish the verified release archive's extracted
crate with `cargo publish --locked` from that directory. Release automation may
use `--no-verify` only when that exact archive has already passed verification.
Keep the compressed `.crate` below the registry's size limit (10 MiB by default).

Other local checks:

```sh
python3 -m unittest discover -s rust/tests -p 'test_package.py'
cargo fmt -- --check
cargo clippy --all-targets -j 3 -- -D warnings
```

## Usage

```rust,no_run
use clap_cpp::CLAP;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = CLAP::new("model.gguf", 4)?;
    let text_embedding = model.encode_text("a piano")?;
    let audio_embedding = model.encode_audio("piano.wav", 0)?;
    Ok(())
}
```
