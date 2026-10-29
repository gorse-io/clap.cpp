# clap.cpp Rust bindings

## Install

```sh
cargo add clap-cpp
```

Requires CMake 3.20 or newer and a C++17 compiler (MSVC on Windows). The
published crate bundles clap.cpp, ggml, JSON, ICU Unicode/regex,
and miniaudio sources. Default builds are CPU-only and do not download dependencies
or require Git, Python, GPU SDKs, shared ICU libraries, or the parent repository. Cross-compiling
is not currently supported. Python 3 is required only for integration-test
fixture generation and maintainer packaging.

## GPU features

Enable an optional backend when adding the dependency:

```sh
cargo add clap-cpp --features cuda
# Or: --features vulkan, --features rocm, --features metal
```

| Feature | Native backend | Build requirements |
| --- | --- | --- |
| `cuda` | GGML CUDA | NVIDIA CUDA Toolkit and a supported C++ host compiler |
| `vulkan` | GGML Vulkan | Vulkan headers/loader, `glslc`, and SPIR-V headers |
| `rocm` | GGML HIP | AMD ROCm/HIP SDK with hipBLAS and rocBLAS |
| `metal` | GGML Metal | Apple SDK and Metal compiler on macOS |

The CPU backend remains available with every feature. Features compile backends;
they do not guarantee that a compatible GPU is present or that every operation
runs on it. Backend selection follows clap.cpp's native behavior, without changes
to the Rust API. CUDA and ROCm cannot be enabled together; do not use
`--all-features` to test this crate. GPU SDKs and runtime drivers are external
requirements and are not bundled in the crate. Enabled backends may introduce
shared vendor runtime dependencies even though clap.cpp and ggml are linked
statically.

For checkout builds, use e.g. `cargo test --manifest-path rust/Cargo.toml
--features vulkan --all-targets -j 3`. SDK discovery uses CMake; configure
nonstandard installations with the SDK's usual environment variables (such as
`VULKAN_SDK` or `HIP_PATH`) or a `CMAKE_TOOLCHAIN_FILE`.

## Building from Git

Initialize the pinned submodules before using a path dependency or building:

```sh
git submodule update --init --recursive
cargo test --manifest-path rust/Cargo.toml --all-targets -j 3
cargo test --manifest-path rust/Cargo.toml --doc -j 3
```

Without `rust/vendor/`, the build uses the parent checkout. If a vendored tree
exists, it takes precedence; regenerate it after changing native sources or
remove `rust/vendor/` to resume building directly from the checkout.

## Packaging and publishing (maintainers)

From the repository root, with Python 3 and the pinned submodules initialized:

```sh
python3 rust/scripts/vendor.py
cargo package --manifest-path rust/Cargo.toml --allow-dirty --locked -j 3
```

`vendor.py` rejects missing, wrong-revision, or dirty submodules before copying
anything. It stages the native build source subsets (including optional GPU
backends), upstream licenses,
root Apache license, submodule revision manifest, and the stdlib-only tiny-model
fixture generator into ignored `rust/vendor/`. The script can be rerun; no
vendored source or generated model/audio fixtures should be committed. Cargo's
explicit include list adds the ignored vendor tree to the archive while omitting
`target/`, compiled binaries, and generated fixture output. The archive includes
all native dependency licenses in their upstream locations beneath `vendor/`.

Do not skip Cargo's package verification: it builds the archived source rather
than the checkout. Also run integration tests against a standalone extraction
(no parent source tree), replacing the version below if needed:

```sh
mkdir -p rust/target/standalone
tar -xzf rust/target/package/clap-cpp-0.1.0.crate -C rust/target/standalone
cargo test --manifest-path rust/target/standalone/clap-cpp-0.1.0/Cargo.toml --all-targets --locked -j 3
cargo test --manifest-path rust/target/standalone/clap-cpp-0.1.0/Cargo.toml --doc --locked -j 3
```

After review and verification, publish the verified release archive's extracted
crate with `cargo publish --locked` from that directory. Release automation may
use `--no-verify` only when that exact archive has already passed verification.
Keep the compressed `.crate` below the registry's size limit (10 MiB by default).

Other local checks:

```sh
python3 -m unittest discover -s rust/tests -p 'test_vendor.py'
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets -j 3 -- -D warnings
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
