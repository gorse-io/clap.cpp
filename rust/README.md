# clap.cpp Rust bindings

[![crates.io](https://img.shields.io/crates/v/clap-cpp)](https://crates.io/crates/clap-cpp)
[![Crates.io Total Downloads](https://img.shields.io/crates/d/clap-cpp)](https://crates.io/crates/clap-cpp)
[![docs.rs](https://img.shields.io/docsrs/clap-cpp)](https://docs.rs/clap-cpp)

Require CMake 3.20 or newer and a C++20 compiler (MSVC on Windows).

## Install

```sh
cargo add clap-cpp
```

| Feature | Native backend | Build requirements |
| --- | --- | --- |
| `cuda` | GGML CUDA | NVIDIA CUDA Toolkit and a supported C++ host compiler |
| `vulkan` | GGML Vulkan | Vulkan headers/loader, `glslc`, and SPIR-V headers |
| `rocm` | GGML HIP | AMD ROCm/HIP SDK with hipBLAS and rocBLAS |

Metal is enabled automatically on macOS.

## Usage

```rust,no_run
use clap_cpp::CLAP;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = CLAP::new("larger_clap_general-q4_k_m.gguf", 4)?;
    let text_embedding = model.encode_text("Sound of a dog")?;
    let audio_embedding = model.encode_audio("dog.wav", 0)?;
    Ok(())
}
```

For EmbeddingGemma 2 audio, load the main GGUF and matching mmproj with
`CLAP::new_with_mmproj("embeddinggemma2.gguf", "mmproj-Q8_0.gguf", 4)?`, then
call `model.encode_audio("recording.wav", 0)?` as above. Both paths accept
`impl AsRef<Path>`; missing/incompatible projectors return errors. The original
`new(path, threads)` remains unchanged, and text does not require an mmproj.
Image encoding is not supported.
