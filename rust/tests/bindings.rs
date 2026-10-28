use clap_cpp::CLAP;
use std::{path::PathBuf, process::Command, sync::OnceLock};

fn fixtures() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("fixtures-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let status = Command::new(if cfg!(windows) { "python" } else { "python3" })
            .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/generate.py"))
            .arg(&dir)
            .status()
            .unwrap();
        assert!(status.success());
        dir
    })
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 2e-5, "{a} != {b}");
    }
}

#[test]
fn errors_are_owned_and_inputs_are_checked() {
    assert!(CLAP::new("missing.gguf", 1).is_err());
    assert!(CLAP::new("nul\0.gguf", 1).is_err());
    assert!(CLAP::new("missing.gguf", -1).is_err());
    let mut model = CLAP::new(fixtures().join("模型.gguf"), 1).unwrap();
    assert!(model.encode_text("hi\0").is_err());
    assert!(model.encode_audio("bad\0.wav", 0).is_err());
    let error = model.encode_audio("missing.wav", 0).unwrap_err();
    let message = error.to_string();
    model.encode_text("ok").unwrap();
    assert!(!message.is_empty());
    assert_eq!(error.to_string(), message);
    assert!(model
        .encode_audio(fixtures().join("invalid.wav"), 0)
        .is_err());
}

#[test]
fn clap_real_text_and_audio_file() {
    let mut model = CLAP::new(fixtures().join("模型.gguf"), 1).unwrap();
    let text = model.encode_text("Hi").unwrap();
    assert_eq!(text.len(), 3);
    let expected: Vec<f32> = std::fs::read_to_string(fixtures().join("text-reference.txt"))
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    close(&text, &expected);
    assert!((text.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-5);
    let audio = model.encode_audio(fixtures().join("音频.wav"), 0).unwrap();
    assert_eq!(audio.len(), 3);
    assert!(audio.iter().all(|v| v.is_finite()));
    assert!((audio.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-5);
    // Stereo 24 kHz input is downmixed internally, then resampled to 48 kHz.
    close(
        &audio,
        &model
            .encode_audio(fixtures().join("立体声.wav"), 0)
            .unwrap(),
    );
    let cropped = model
        .encode_audio(fixtures().join("long.wav"), 48000)
        .unwrap();
    close(
        &cropped,
        &model.encode_audio(fixtures().join("crop.wav"), 0).unwrap(),
    );
}

#[test]
fn mert_audio_file_and_unsupported_operations() {
    let mut model = CLAP::new(fixtures().join("mert.gguf"), 1).unwrap();
    let audio = model.encode_audio(fixtures().join("mert.wav"), 0).unwrap();
    assert_eq!(audio.len(), 4);
    assert!(audio.iter().all(|v| v.is_finite()));
    assert!(model.encode_text("not supported").is_err());
    assert!(model.encode_audio(fixtures().join("mert.wav"), 1).is_err());
}

#[cfg(unix)]
#[test]
fn non_utf8_paths_are_rejected() {
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt};
    let invalid = std::path::Path::new(OsStr::from_bytes(b"invalid\xff.wav"));
    assert!(CLAP::new(invalid, 1).is_err());
    let mut model = CLAP::new(fixtures().join("模型.gguf"), 1).unwrap();
    assert!(model.encode_audio(invalid, 0).is_err());
}
