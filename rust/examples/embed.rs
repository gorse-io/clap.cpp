use clap_cpp::{decode_audio, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: cargo run --example embed -- MODEL.gguf TEXT [AUDIO]")?;
    let text = args.next().ok_or("missing text")?;
    let mut model = Model::new(path, 0)?;
    println!("projection dimension: {}", model.projection_dimension()?);
    println!("text: {:?}", model.text(&text)?);
    if let Some(path) = args.next() {
        let audio = decode_audio(path)?;
        println!(
            "audio: {:?}",
            model.audio(&audio.samples, audio.sample_rate, 0)?
        );
    }
    Ok(())
}
