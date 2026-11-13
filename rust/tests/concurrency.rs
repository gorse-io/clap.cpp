//! Offline concurrency regressions. Optional CLAP_TEST_MODEL must support both
//! text and waveform audio; CLAP_TEST_AUDIO can supply its matching input file.
//! Otherwise tests use the repository's stdlib-only tiny waveform model generator.
use clap_cpp::CLAP;
use std::{
    ffi::{c_char, CStr},
    path::PathBuf,
    process::Command,
    sync::{Arc, Barrier, OnceLock},
    thread,
};

struct Fixtures {
    model: PathBuf,
    audio: PathBuf,
}

fn fixtures() -> &'static Fixtures {
    static FIXTURES: OnceLock<Fixtures> = OnceLock::new();
    FIXTURES.get_or_init(|| {
        let root = std::env::var_os("CLAP_TEST_ARTIFACTS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("target")
                    .join(format!("rust-concurrency-{}", std::process::id()))
            });
        std::fs::create_dir_all(&root).unwrap();
        let model = std::env::var_os("CLAP_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("tiny-waveform.gguf"));
        let audio = std::env::var_os("CLAP_TEST_AUDIO")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("tone.wav"));
        let script = r#"
import importlib.util, math, struct, sys, wave
sys.dont_write_bytecode = True
model, audio, generator, generate_model, generate_audio = sys.argv[1:]
if generate_model == '1':
    spec = importlib.util.spec_from_file_location('tiny_model', generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_model(model, waveform=True)
if generate_audio == '1':
    with wave.open(audio, 'wb') as output:
        output.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
        output.writeframes(b''.join(struct.pack('<h', int(6000 * math.sin(2 * math.pi * 440 * i / 48000))) for i in range(48000)))
"#;
        let python = std::env::var_os("PYTHON")
            .unwrap_or_else(|| if cfg!(windows) { "python" } else { "python3" }.into());
        let status = Command::new(python)
            .arg("-c")
            .arg(script)
            .arg(&model)
            .arg(&audio)
            .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/tiny_model.py"))
            .arg(if std::env::var_os("CLAP_TEST_MODEL").is_some() { "0" } else { "1" })
            .arg(if std::env::var_os("CLAP_TEST_AUDIO").is_some() { "0" } else { "1" })
            .status()
            .expect("run stdlib-only fixture generator (or set PYTHON)");
        assert!(status.success(), "fixture generation failed");
        assert!(model.is_file(), "missing model: {}", model.display());
        assert!(audio.is_file(), "missing audio: {}", audio.display());
        Fixtures { model, audio }
    })
}

fn model_is_send_sync<T: Send + Sync>() {}

#[test]
fn shared_reference_api_is_send_sync() {
    model_is_send_sync::<CLAP>();
    // Type-check both methods without loading a model or calling native code.
    let _text: fn(&CLAP, &str) -> _ = CLAP::encode_text;
    let _audio: fn(&CLAP, PathBuf, usize) -> _ = |model, path, crop| model.encode_audio(path, crop);
}

fn assert_embedding(actual: &[f32], expected: &[f32]) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 1e-5, "embedding differs: {a} vs {b}");
    }
}

#[test]
fn arc_model_encodes_text_and_audio_concurrently() {
    let fixture = fixtures();
    let model = Arc::new(CLAP::new(&fixture.model, 1).unwrap());
    let expected_text = model.encode_text("a steady tone").unwrap();
    let expected_audio = model.encode_audio(&fixture.audio, 0).unwrap();
    let start = Arc::new(Barrier::new(4));
    let workers: Vec<_> = (0..4)
        .map(|index| {
            let model = Arc::clone(&model);
            let start = Arc::clone(&start);
            let audio = fixture.audio.clone();
            thread::spawn(move || {
                start.wait();
                (0..3)
                    .map(|_| {
                        if index % 2 == 0 {
                            model.encode_text("a steady tone").unwrap()
                        } else {
                            model.encode_audio(&audio, 0).unwrap()
                        }
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    // Only worker Arcs keep the native handle alive during inference.
    drop(model);
    for (index, worker) in workers.into_iter().enumerate() {
        for result in worker.join().unwrap() {
            assert_embedding(
                &result,
                if index % 2 == 0 {
                    &expected_text
                } else {
                    &expected_audio
                },
            );
        }
    }
}

#[test]
fn embeddings_outlive_the_model() {
    let fixture = fixtures();
    let (text, audio) = {
        let model = CLAP::new(&fixture.model, 1).unwrap();
        (
            model.encode_text("a steady tone").unwrap(),
            model.encode_audio(&fixture.audio, 0).unwrap(),
        )
    };
    // Read and copy both allocations only after the native handle is destroyed.
    assert_embedding(&text, &text.clone());
    assert_embedding(&audio, &audio.clone());
}

unsafe extern "C" {
    fn clap_last_error() -> *const c_char;
}

fn native_diagnostic() -> String {
    // SAFETY: The C ABI returns a NUL-terminated thread-local diagnostic. No
    // fallible native call intervenes before this copy on the calling thread.
    unsafe { CStr::from_ptr(clap_last_error()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn native_failures_and_successes_keep_thread_local_diagnostics() {
    let model = Arc::new(CLAP::new(&fixtures().model, 1).unwrap());
    let rendezvous = Arc::new(Barrier::new(2));
    let workers: Vec<_> = (0..2)
        .map(|index| {
            let model = Arc::clone(&model);
            let rendezvous = Arc::clone(&rendezvous);
            thread::spawn(move || {
                let mut observations = Vec::new();
                for _ in 0..8 {
                    // Distinct native errors, not CString/path validation errors.
                    let input = if index == 0 {
                        ""
                    } else {
                        "missing-concurrency-audio.wav"
                    };
                    let failure = model.encode_audio(input, 0);
                    let error = failure.as_ref().err().map(ToString::to_string);
                    rendezvous.wait();
                    observations.push(error.as_deref() == Some(native_diagnostic().as_str()));
                    rendezvous.wait();
                    if index == 0 {
                        observations.push(model.encode_text("a steady tone").is_ok());
                        observations.push(native_diagnostic().is_empty());
                    }
                    rendezvous.wait();
                    if index == 1 {
                        observations.push(error.as_deref() == Some(native_diagnostic().as_str()));
                    }
                    rendezvous.wait();
                }
                // Assert after all rendezvous so a TLS regression cannot strand
                // the other worker behind a barrier and hang the test.
                assert!(
                    observations.into_iter().all(|valid| valid),
                    "native thread-local diagnostic was lost or overwritten"
                );
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}
