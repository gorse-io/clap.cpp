# clap.cpp Rust bindings

[![crates.io](https://img.shields.io/crates/v/clap-cpp)](https://crates.io/crates/clap-cpp)
[![Crates.io Total Downloads](https://img.shields.io/crates/d/clap-cpp)](https://crates.io/crates/clap-cpp)
[![docs.rs](https://img.shields.io/docsrs/clap-cpp)](https://docs.rs/clap-cpp)


## Install

Rust bindings require CMake 3.20 or newer and a C++20 compiler (MSVC on Windows).

```sh
cargo add clap-cpp
```

## GPU features

Enable an optional backend when adding the dependency (Metal is already enabled
by default on macOS):

```sh
cargo add clap-cpp --features cuda|vulkan|rocm|metal
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

From the repository root, with the pinned submodules initialized:

```sh
cargo package --locked -j 3
```

Use clean submodules at the revisions pinned by the checkout. Cargo includes the
initialized Git submodules directly through the root manifest's include allowlist: the native
CMake graph, clap.cpp sources, selected dependency source trees (including optional
GPU backends), upstream licenses in their original locations, and
`python/tests/tiny_models.py` for standalone tests. There is no copying step and
no `vendor/` tree. Unrelated CLI dependencies, generated models, audio, build
output and Git metadata are omitted. Do not commit generated fixtures or archives.
For an intentional dirty local package use `--allow-dirty`; releases should use
the pinned clean source tree.

Do not skip Cargo's package verification: it builds the archived source rather
than the checkout. Run integration and documentation tests from the Git checkout
with `cargo test --all-targets` and `cargo test --doc`, not from the packaged crate.

After review and verification, publish the verified release archive's extracted
crate with `cargo publish --locked` from that directory. Release automation may
use `--no-verify` only when that exact archive has already passed verification.
Keep the compressed `.crate` below the registry's size limit (10 MiB by default).

Other local checks:

```sh
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
