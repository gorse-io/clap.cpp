//! Safe, owning CPU bindings with the Python-style CLAP encoding API.
//!
//! Build from a recursive Git checkout; see `rust/README.md`. Embeddings are
//! Rust-owned vectors. A CLAP instance can be shared across threads, including
//! concurrent text and audio encoding, using `std::sync::Arc`.
use std::{
    ffi::{c_char, c_int, c_void, CStr, CString},
    path::Path,
    ptr::NonNull,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// The small public C11 ABI from include/clap.h. Keep native implementation
// details out of this crate; no generated bindings or C++ adapter are needed.
mod ffi {
    use super::{c_char, c_int, c_void};

    #[repr(C)]
    pub struct Model {
        _private: [u8; 0],
    }

    // C enums use c_int, not Rust enums: unknown native discriminants must not
    // produce an invalid Rust value at the FFI boundary.
    pub type Status = c_int;
    pub const OK: Status = 0;

    #[repr(C)]
    pub struct RuntimeOptions {
        pub threads: c_int,
        pub device: *const c_char,
        pub activation_precision: c_int,
    }

    unsafe extern "C" {
        pub fn clap_default_options() -> RuntimeOptions;
        pub fn clap_last_error() -> *const c_char;
        pub fn clap_create(
            path: *const c_char,
            options: *const RuntimeOptions,
            out: *mut *mut Model,
        ) -> Status;
        pub fn clap_destroy(model: *mut Model);
        pub fn clap_encode_text(
            model: *mut Model,
            text: *const c_char,
            values: *mut *mut f32,
            count: *mut usize,
        ) -> Status;
        pub fn clap_encode_audio_file(
            model: *mut Model,
            path: *const c_char,
            crop_offset: usize,
            values: *mut *mut f32,
            count: *mut usize,
        ) -> Status;
        pub fn clap_free(allocation: *mut c_void);
    }
}

#[derive(Default)]
struct NativeBuffer {
    values: *mut f32,
    count: usize,
}

impl Drop for NativeBuffer {
    fn drop(&mut self) {
        // SAFETY: The pointer is null or an allocation returned by clap_*.
        // clap_free accepts null and is the matching allocator's release API.
        unsafe { ffi::clap_free(self.values.cast()) }
    }
}

impl NativeBuffer {
    fn into_vec(self) -> Result<Vec<f32>> {
        // Even a zero-length Rust slice requires a nonnull, aligned pointer.
        if self.count == 0 {
            return Ok(Vec::new());
        }
        if self.values.is_null() {
            return Err("native embedding pointer is null".into());
        }
        if self.count > isize::MAX as usize / std::mem::size_of::<f32>() {
            return Err("native embedding length exceeds Rust slice limits".into());
        }
        // SAFETY: Success supplies count initialized, aligned floats in one
        // native allocation. The guard lives through to_vec (including unwind)
        // and releases it only after Rust has copied its independent vector.
        Ok(unsafe { std::slice::from_raw_parts(self.values, self.count) }.to_vec())
    }
}

fn path(p: &Path) -> Result<CString> {
    Ok(CString::new(p.to_str().ok_or("path must be valid UTF-8")?)?)
}

fn check_status(status: ffi::Status) -> Result<()> {
    if status == ffi::OK {
        return Ok(());
    }
    // SAFETY: The API provides a thread-local NUL-terminated diagnostic valid
    // until the next fallible call. Copy it immediately, before any such call.
    let diagnostic = unsafe { ffi::clap_last_error() };
    let message = if diagnostic.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(diagnostic) }
            .to_string_lossy()
            .into_owned()
    };
    if message.is_empty() {
        Err(format!("native call failed with status {status}").into())
    } else {
        Err(message.into())
    }
}

/// Owned shared immutable native model, selected from the GGUF metadata.
/// Drop releases weights and all native resources.
///
/// Encoding borrows the model and uses private native execution state per call.
/// Share an instance with `std::sync::Arc`; its last owner releases the model
/// only after all borrowed calls finish. Returned vectors outlive the model.
/// ```
/// fn assert_send_sync<T: Send + Sync>() {}
/// assert_send_sync::<clap_cpp::CLAP>();
/// ```
pub struct CLAP {
    ptr: NonNull<ffi::Model>,
}

// SAFETY: The C handle owns immutable shared weights and runtime options, not
// thread-affine execution state. Each encode call creates its own private native
// Context, and diagnostics are thread-local. Moving ownership to another thread
// is supported by the native API, including destruction on that thread.
unsafe impl Send for CLAP {}

// SAFETY: Concurrent native calls on the same model are supported: execution
// state and output buffers are private to each call. Methods require a live
// &self, so Rust borrowing (also through Arc) prevents Drop racing an active
// call. The handle is private and is never exposed for external destruction.
unsafe impl Sync for CLAP {}

impl Drop for CLAP {
    fn drop(&mut self) {
        // SAFETY: This instance uniquely owns the handle from clap_create;
        // Drop's exclusive access means all borrowed calls have finished.
        unsafe { ffi::clap_destroy(self.ptr.as_ptr()) }
    }
}
impl CLAP {
    /// Load a UTF-8 GGUF path. Zero threads selects the native default.
    pub fn new(p: impl AsRef<Path>, threads: i32) -> Result<Self> {
        let s = path(p.as_ref())?;
        if threads < 0 {
            return Err("threads must be nonnegative".into());
        }
        // SAFETY: No arguments; the C ABI returns this repr(C) struct by value.
        let mut options = unsafe { ffi::clap_default_options() };
        options.threads = threads;
        let mut created = std::ptr::null_mut();
        // SAFETY: Strings/options live through the synchronous call; created
        // is a fresh writable slot. On failure the API leaves it null.
        let status = unsafe { ffi::clap_create(s.as_ptr(), &options, &mut created) };
        check_status(status)?;
        let ptr = NonNull::new(created).ok_or("native model missing")?;
        Ok(Self { ptr })
    }
    /// Encode one UTF-8 string using the model's native context policy.
    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        let s = CString::new(text)?;
        let mut output = NativeBuffer::default();
        // SAFETY: The shared borrow keeps the immutable model live throughout
        // this synchronous call, which owns private native execution state.
        // The string stays valid and the guard owns the fresh output slots.
        let status = unsafe {
            ffi::clap_encode_text(
                self.ptr.as_ptr(),
                s.as_ptr(),
                &mut output.values,
                &mut output.count,
            )
        };
        check_status(status)?;
        output.into_vec()
    }
    /// Encode a WAV, MP3 or FLAC file. Channels are averaged to mono and the
    /// source sample rate is converted internally to the model's input rate.
    /// `crop_offset` counts model-rate samples (normally 0); architectures that
    /// do not support cropping reject nonzero offsets.
    pub fn encode_audio(&self, p: impl AsRef<Path>, crop_offset: usize) -> Result<Vec<f32>> {
        let s = path(p.as_ref())?;
        let mut output = NativeBuffer::default();
        // SAFETY: As for encode_text; crop_offset uses C size_t (Rust usize).
        let status = unsafe {
            ffi::clap_encode_audio_file(
                self.ptr.as_ptr(),
                s.as_ptr(),
                crop_offset,
                &mut output.values,
                &mut output.count,
            )
        };
        check_status(status)?;
        output.into_vec()
    }
}
