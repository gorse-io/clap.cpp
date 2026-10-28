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
