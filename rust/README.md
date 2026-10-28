# clap.cpp Rust bindings

## Install

```sh
cargo add clap-cpp
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

For EmbeddingGemma 2 audio, load the main GGUF and matching mmproj with
`CLAP::new_with_mmproj("embeddinggemma2.gguf", "mmproj-Q8_0.gguf", 4)?`, then
call `model.encode_audio("recording.wav", 0)?` as above. Both paths accept
`impl AsRef<Path>`; missing/incompatible projectors return errors. The original
`new(path, threads)` remains unchanged, and text does not require an mmproj.
Image encoding is not supported.
