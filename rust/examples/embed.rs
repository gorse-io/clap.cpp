use clap_cpp::CLAP;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: cargo run --example embed -- MODEL.gguf TEXT [AUDIO]")?;
    let text = args.next().ok_or("missing text")?;
    let mut model = CLAP::new(path, 0)?;
    println!("text: {:?}", model.encode_text(&text)?);
    if let Some(path) = args.next() {
        println!("audio: {:?}", model.encode_audio(path, 0)?);
    }
    Ok(())
}
