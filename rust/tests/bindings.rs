use clap_cpp::{decode_audio, log_mel, resample_audio, Model};
use std::{path::PathBuf, process::Command, sync::OnceLock};

fn fixtures() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("clap-rust-{}", std::process::id()));
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

#[test]
fn errors_are_owned_and_inputs_are_checked() {
    assert!(Model::new("missing.gguf", 1).is_err());
    assert!(Model::new("nul\0.gguf", 1).is_err());
    assert!(Model::new("missing.gguf", -1).is_err());
    assert!(decode_audio("missing.wav").is_err());
    assert!(decode_audio("bad\0.wav").is_err());
    for bad in [&[][..], &[f32::NAN][..], &[f32::INFINITY][..]] {
        assert!(resample_audio(bad, 48000, 24000).is_err());
        assert!(log_mel(bad, 48000, 0).is_err());
    }
    assert!(resample_audio(&[0.], 0, 48000).is_err());
    assert!(resample_audio(&[0.], 48000, -1).is_err());
    assert!(log_mel(&[0.], 24000, 0).is_err());
}

#[test]
fn decode_resample_and_mel_values() {
    let audio = decode_audio(fixtures().join("音频.wav")).unwrap();
    assert_eq!(audio.sample_rate, 24000);
    assert_eq!(audio.samples, vec![0., 0.25, -0.5, 0.75]);
    assert_eq!(
        resample_audio(&audio.samples, 24000, 24000).unwrap(),
        audio.samples
    );
    assert_eq!(
        resample_audio(&audio.samples, 24000, 48000).unwrap().len(),
        8
    );
    let mel = log_mel(&[0.; 480], 48000, 0).unwrap();
    assert_eq!(mel.len(), 1001 * 64);
    assert!(mel
        .iter()
        .all(|x| x.is_finite() && (*x + 100.).abs() < 1e-3));
}

#[test]
fn clap_real_text_and_waveform() {
    let mut model = Model::new(fixtures().join("模型.gguf"), 1).unwrap();
    assert_eq!(model.audio_sample_rate().unwrap(), 48000);
    assert_eq!(model.projection_dimension().unwrap(), 3);
    let ids = model.tokenize("Hi", 8).unwrap();
    assert_eq!(ids, vec![0, 77, 110, 2]);
    let text = model.text("Hi").unwrap();
    assert_eq!(text.len(), 3);
    assert!(text.iter().all(|v| v.is_finite()));
    let expected: Vec<f32> = std::fs::read_to_string(fixtures().join("text-reference.txt"))
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(expected.len(), 3);
    for (a, b) in text.iter().zip(expected) {
        assert!((a - b).abs() < 2e-5, "{a} != {b}");
    }
    let from_tokens = model
        .text_tokens(&model.tokenize("Hi", 512).unwrap())
        .unwrap();
    for (a, b) in text.iter().zip(from_tokens) {
        assert!((a - b).abs() < 1e-5);
    }
    assert!((text.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-5);
    assert!(model.tokenize("hi\0", 8).is_err());
    assert!(model.tokenize("hi", 0).is_err());
    assert!(model.text("hi\0").is_err());
    assert!(model.text_tokens(&[]).is_err());
    assert!(model.text_tokens(&[-1]).is_err());
    assert!(model.text_tokens(&[99999]).is_err());
    assert!(model.audio(&[f32::NAN], 48000, 0).is_err());
    let pcm: Vec<_> = (0..2400).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();
    let audio = model.audio(&pcm, 24000, 0).unwrap();
    assert_eq!(audio.len(), 3);
    assert!(audio.iter().all(|v| v.is_finite()));
    assert!((audio.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-5);
    assert!(model.audio_frames(&pcm, 24000).is_err());
    assert!(model.clamp3_features(&pcm, 24000).is_err());
    // Each error owns its diagnostic; later native calls must not overwrite it.
    let error = model.text_tokens(&[]).unwrap_err().to_string();
    model.text("ok").unwrap();
    assert!(!error.is_empty());
}

#[test]
fn mert_frames_features_and_pooling() {
    let mut model = Model::new(fixtures().join("mert.gguf"), 1).unwrap();
    assert_eq!(model.audio_sample_rate().unwrap(), 24000);
    assert_eq!(model.projection_dimension().unwrap(), 4);
    let pcm: Vec<_> = (0..24000).map(|i| (i as f32 * 0.03).sin() * 0.2).collect();
    let frames = model.audio_frames(&pcm, 24000).unwrap();
    assert_eq!(frames.dimension, 4);
    assert_eq!(frames.frames, 75);
    assert_eq!(frames.values.len(), frames.frames * frames.dimension);
    let pooled = model.audio(&pcm, 24000, 0).unwrap();
    for (i, value) in pooled.iter().enumerate() {
        let mean = frames
            .values
            .chunks_exact(frames.dimension)
            .map(|row| row[i])
            .sum::<f32>()
            / frames.frames as f32;
        assert!((mean - value).abs() < 1e-5);
    }
    let features = model.clamp3_features(&pcm, 24000).unwrap();
    assert_eq!((features.frames, features.dimension), (1, 4));
    assert!(features.values.iter().all(|v| v.is_finite()));
    assert!(model.text("not supported").is_err());
    assert!(model.audio_frames(&[], 24000).is_err());
    assert!(model.clamp3_features(&[0.], 24000).is_err());
}
