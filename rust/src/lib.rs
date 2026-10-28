//! Safe, owning CPU bindings with the Python-style CLAP encoding API.
//!
//! Build from a recursive Git checkout; see `rust/README.md`. Embeddings are
//! Rust-owned vectors. A CLAP instance is deliberately neither Send nor Sync.
use std::{
    ffi::{c_char, c_void, CStr, CString},
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

mod ffi {
    use super::*;
    unsafe extern "C" {
        pub fn cr_call(
            op: i32,
            model: *mut c_void,
            s: *const c_char,
            threads: i32,
            crop_offset: usize,
            mmproj: *const c_char,
        ) -> *mut c_void;
        pub fn cr_free(p: *mut c_void);
        pub fn cr_model_free(p: *mut c_void);
        pub fn cr_model_take(p: *mut c_void) -> *mut c_void;
        pub fn cr_error(p: *const c_void) -> *const c_char;
        pub fn cr_floats(p: *const c_void) -> *const f32;
        pub fn cr_len(p: *const c_void) -> usize;
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
        // Empty C++ vector::data() may be null; never form a slice from it.
        unsafe {
            let n = ffi::cr_len(self.ptr());
            if n == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(ffi::cr_floats(self.ptr()), n).to_vec()
            }
        }
    }
}
fn path(p: &Path) -> Result<CString> {
    Ok(CString::new(p.to_str().ok_or("path must be valid UTF-8")?)?)
}
// The synchronous bridge borrows the string and owns all returned data.
fn call(
    op: i32,
    model: *mut c_void,
    s: &CStr,
    threads: i32,
    crop_offset: usize,
    mmproj: Option<&CStr>,
) -> Result<Native> {
    let p = unsafe {
        ffi::cr_call(
            op,
            model,
            s.as_ptr(),
            threads,
            crop_offset,
            mmproj.map_or(std::ptr::null(), CStr::as_ptr),
        )
    };
    let result = Native(NonNull::new(p).ok_or("native result allocation failed")?);
    let error = unsafe { CStr::from_ptr(ffi::cr_error(result.ptr())) };
    if !error.is_empty() {
        return Err(error.to_string_lossy().into_owned().into());
    }
    Ok(result)
}

/// Owned native model, selected from the GGUF architecture metadata.
/// Drop releases weights and all native resources.
///
/// Inference requires exclusive access: the native backend does not guarantee
/// same-instance thread safety.
/// ```compile_fail
/// fn send<T: Send>() {}
/// send::<clap_cpp::CLAP>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {}
/// sync::<clap_cpp::CLAP>();
/// ```
pub struct CLAP {
    ptr: NonNull<c_void>,
    _not_send_sync: PhantomData<Rc<()>>,
}
impl Drop for CLAP {
    fn drop(&mut self) {
        unsafe { ffi::cr_model_free(self.ptr.as_ptr()) }
    }
}
impl CLAP {
    /// Load a UTF-8 GGUF path. Zero threads selects the native default.
    pub fn new(p: impl AsRef<Path>, threads: i32) -> Result<Self> {
        Self::load(p.as_ref(), None, threads)
    }
    /// Load a model with a separate audio mmproj GGUF (EmbeddingGemma2).
    /// Both paths must be UTF-8 without NUL; zero threads selects the native default.
    pub fn new_with_mmproj(
        p: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        threads: i32,
    ) -> Result<Self> {
        Self::load(p.as_ref(), Some(mmproj.as_ref()), threads)
    }
    fn load(p: &Path, mmproj: Option<&Path>, threads: i32) -> Result<Self> {
        let s = path(p)?;
        let audio_path = mmproj.map(path).transpose()?;
        if audio_path.as_ref().is_some_and(|p| p.as_bytes().is_empty()) {
            return Err("mmproj path must be nonempty".into());
        }
        let native = call(
            0,
            std::ptr::null_mut(),
            &s,
            threads,
            0,
            audio_path.as_deref(),
        )?;
        let ptr = NonNull::new(unsafe { ffi::cr_model_take(native.ptr()) })
            .ok_or("native model missing")?;
        Ok(Self {
            ptr,
            _not_send_sync: PhantomData,
        })
    }
    /// Encode one UTF-8 string using the model's native context policy.
    pub fn encode_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let s = CString::new(text)?;
        Ok(call(1, self.ptr.as_ptr(), &s, 0, 0, None)?.floats())
    }
    /// Encode a WAV, MP3 or FLAC file. Channels are averaged to mono and the
    /// source sample rate is converted internally to the model's input rate.
    /// `crop_offset` counts model-rate samples (normally 0); architectures that
    /// do not support cropping reject nonzero offsets.
    pub fn encode_audio(&mut self, p: impl AsRef<Path>, crop_offset: usize) -> Result<Vec<f32>> {
        let s = path(p.as_ref())?;
        Ok(call(2, self.ptr.as_ptr(), &s, 0, crop_offset, None)?.floats())
    }
}
