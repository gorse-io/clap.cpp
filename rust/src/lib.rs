//! Safe, self-contained CPU bindings to the clap.cpp C++ API.
//!
//! Build from a recursive Git checkout; see `rust/README.md`. Native results are
//! copied into Rust-owned vectors. A model is deliberately neither Send nor Sync.
use std::{
    ffi::{c_char, c_void, CStr, CString},
    fmt,
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

/// An owned diagnostic from argument validation or native inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
/// Fallible binding operation.
pub type Result<T> = std::result::Result<T, Error>;

mod ffi {
    use super::*;
    unsafe extern "C" {
        pub fn cr_call(
            op: i32,
            model: *mut c_void,
            s: *const c_char,
            f: *const f32,
            ids: *const i32,
            n: usize,
            rate: i32,
            extra: usize,
        ) -> *mut c_void;
        pub fn cr_free(p: *mut c_void);
        pub fn cr_model_free(p: *mut c_void);
        pub fn cr_model_take(p: *mut c_void) -> *mut c_void;
        pub fn cr_error(p: *const c_void) -> *const c_char;
        pub fn cr_floats(p: *const c_void) -> *const f32;
        pub fn cr_ints(p: *const c_void) -> *const i32;
        pub fn cr_len(p: *const c_void, ints: i32) -> usize;
        pub fn cr_meta(p: *const c_void, field: i32) -> i32;
    }
}
struct Native(NonNull<c_void>);
impl Drop for Native {
    fn drop(&mut self) {
        unsafe { ffi::cr_free(self.0.as_ptr()) }
    }
}
impl Native {
    fn ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }
    fn floats(&self) -> Vec<f32> {
        // C++ owns a contiguous vector until this Native is dropped. Never form
        // a Rust slice from vector::data() when empty (it may be null).
        unsafe {
            let n = ffi::cr_len(self.ptr(), 0);
            if n == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(ffi::cr_floats(self.ptr()), n).to_vec()
            }
        }
    }
    fn meta(&self, field: i32) -> i32 {
        unsafe { ffi::cr_meta(self.ptr(), field) }
    }
    fn frames(&self) -> Result<AudioFrames> {
        let frames =
            usize::try_from(self.meta(0)).map_err(|_| Error("negative frame count".into()))?;
        let dimension =
            usize::try_from(self.meta(1)).map_err(|_| Error("negative frame dimension".into()))?;
        let values = self.floats();
        if frames.checked_mul(dimension) != Some(values.len()) {
            return Err(Error("invalid native frame shape".into()));
        }
        Ok(AudioFrames {
            values,
            frames,
            dimension,
        })
    }
}
fn string(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error("interior NUL is not allowed".into()))
}
fn path(p: &Path) -> Result<CString> {
    string(
        p.to_str()
            .ok_or_else(|| Error("path must be valid UTF-8".into()))?,
    )
}
// All pointer/length pairs are constructed from live borrowed slices here, and
// the synchronous bridge copies them before returning. No raw pointers escape.
#[allow(clippy::too_many_arguments)]
fn call(
    op: i32,
    model: *mut c_void,
    s: Option<&CStr>,
    pcm: &[f32],
    ids: &[i32],
    rate: i32,
    extra: usize,
) -> Result<Native> {
    let n = if op == 3 { ids.len() } else { pcm.len() };
    let p = unsafe {
        ffi::cr_call(
            op,
            model,
            s.map_or(std::ptr::null(), CStr::as_ptr),
            pcm.as_ptr(),
            ids.as_ptr(),
            n,
            rate,
            extra,
        )
    };
    let result =
        Native(NonNull::new(p).ok_or_else(|| Error("native result allocation failed".into()))?);
    let error = unsafe { CStr::from_ptr(ffi::cr_error(result.ptr())) };
    if !error.is_empty() {
        return Err(Error(error.to_string_lossy().into_owned()));
    }
    Ok(result)
}

/// Mono floating-point PCM and its source sample rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: i32,
}
/// Time-major frames; values has exactly `frames * dimension` elements.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFrames {
    pub values: Vec<f32>,
    pub frames: usize,
    pub dimension: usize,
}

/// Owned native model. Drop frees weights and all native resources.
///
/// Inference requires exclusive access and models are deliberately !Send/!Sync:
/// the native backend does not guarantee same-instance thread safety.
/// ```compile_fail
/// fn send<T: Send>() {}
/// send::<clap_cpp::Model>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {}
/// sync::<clap_cpp::Model>();
/// ```
pub struct Model {
    ptr: NonNull<c_void>,
    _not_send_sync: PhantomData<Rc<()>>,
}
impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ffi::cr_model_free(self.ptr.as_ptr()) }
    }
}
impl Model {
    /// Load a UTF-8 GGUF path. Zero threads uses the native default.
    pub fn new(p: impl AsRef<Path>, threads: i32) -> Result<Self> {
        let s = path(p.as_ref())?;
        let native = call(0, std::ptr::null_mut(), Some(&s), &[], &[], threads, 0)?;
        let ptr = NonNull::new(unsafe { ffi::cr_model_take(native.ptr()) })
            .ok_or_else(|| Error("native model missing".into()))?;
        Ok(Self {
            ptr,
            _not_send_sync: PhantomData,
        })
    }
    /// Tokenize UTF-8 up to max_length using the model tokenizer, without padding.
    pub fn tokenize(&self, text: &str, max_length: usize) -> Result<Vec<i32>> {
        let s = string(text)?;
        let native = call(1, self.ptr.as_ptr(), Some(&s), &[], &[], 0, max_length)?;
        unsafe {
            let n = ffi::cr_len(native.ptr(), 1);
            Ok(if n == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(ffi::cr_ints(native.ptr()), n).to_vec()
            })
        }
    }
    /// Embed one string with the model's native string/context policy.
    pub fn text(&mut self, text: &str) -> Result<Vec<f32>> {
        let s = string(text)?;
        Ok(call(2, self.ptr.as_ptr(), Some(&s), &[], &[], 0, 0)?.floats())
    }
    /// Embed caller-supplied token IDs. Invalid IDs/context lengths are errors.
    pub fn text_tokens(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        Ok(call(3, self.ptr.as_ptr(), None, &[], tokens, 0, 0)?.floats())
    }
    /// Embed finite mono PCM; the native model resamples automatically.
    /// Crop offsets count samples in the model's preprocessed input domain.
    pub fn audio(&mut self, pcm: &[f32], sample_rate: i32, crop_offset: usize) -> Result<Vec<f32>> {
        Ok(call(
            4,
            self.ptr.as_ptr(),
            None,
            pcm,
            &[],
            sample_rate,
            crop_offset,
        )?
        .floats())
    }
    /// Unpooled final-layer frame sequence (unsupported models return errors).
    pub fn audio_frames(&mut self, pcm: &[f32], sample_rate: i32) -> Result<AudioFrames> {
        call(5, self.ptr.as_ptr(), None, pcm, &[], sample_rate, 0)?.frames()
    }
    /// MERT-v1 all-layer, five-second-window CLaMP3 frontend features.
    pub fn clamp3_features(&mut self, pcm: &[f32], sample_rate: i32) -> Result<AudioFrames> {
        call(6, self.ptr.as_ptr(), None, pcm, &[], sample_rate, 0)?.frames()
    }
    /// Native waveform input rate.
    pub fn audio_sample_rate(&self) -> Result<i32> {
        Ok(call(7, self.ptr.as_ptr(), None, &[], &[], 0, 0)?.meta(2))
    }
    /// Width of the model's embedding output.
    pub fn projection_dimension(&self) -> Result<i32> {
        Ok(call(7, self.ptr.as_ptr(), None, &[], &[], 0, 0)?.meta(1))
    }
}
/// Decode WAV/MP3/FLAC by contents, averaging channels; preserves source rate.
pub fn decode_audio(p: impl AsRef<Path>) -> Result<Audio> {
    let s = path(p.as_ref())?;
    let native = call(8, std::ptr::null_mut(), Some(&s), &[], &[], 0, 0)?;
    Ok(Audio {
        samples: native.floats(),
        sample_rate: native.meta(2),
    })
}
/// Native linear/anti-alias resampler; output length is ceil(n * target/source).
/// Same-rate samples are unchanged; nonempty finite mono input is required.
pub fn resample_audio(pcm: &[f32], source_rate: i32, target_rate: i32) -> Result<Vec<f32>> {
    let rate =
        usize::try_from(target_rate).map_err(|_| Error("target rate must be positive".into()))?;
    Ok(call(9, std::ptr::null_mut(), None, pcm, &[], source_rate, rate)?.floats())
}
/// 48 kHz mono PCM to flattened time-major [1001, 64] Slaney log-power mel.
/// Short inputs use native repeat-padding; crop_offset is in 48 kHz samples.
pub fn log_mel(pcm: &[f32], sample_rate: i32, crop_offset: usize) -> Result<Vec<f32>> {
    Ok(call(
        10,
        std::ptr::null_mut(),
        None,
        pcm,
        &[],
        sample_rate,
        crop_offset,
    )?
    .floats())
}
