// Private FFI contract regressions, compiled as the crate's unit tests.
use super::*;
use std::mem::{align_of, offset_of, size_of};

#[test]
fn runtime_options_match_c_layout_and_by_value_defaults() {
    assert_eq!(size_of::<ffi::Status>(), size_of::<std::ffi::c_int>());
    assert_eq!(offset_of!(ffi::RuntimeOptions, threads), 0);
    let pointer_offset = size_of::<std::ffi::c_int>().next_multiple_of(align_of::<*const c_char>());
    assert_eq!(offset_of!(ffi::RuntimeOptions, device), pointer_offset);
    assert_eq!(
        offset_of!(ffi::RuntimeOptions, activation_precision),
        pointer_offset + size_of::<*const c_char>()
    );
    assert_eq!(
        align_of::<ffi::RuntimeOptions>(),
        align_of::<*const c_char>()
    );
    // A real C struct return exercises the ABI, not a Rust-created approximation.
    let defaults = unsafe { ffi::clap_default_options() };
    assert_eq!(defaults.threads, 0);
    assert!(defaults.device.is_null());
    assert_eq!(defaults.activation_precision, 0);
    let _: unsafe extern "C" fn(
        *mut ffi::Model,
        *const c_char,
        usize,
        *mut *mut f32,
        *mut usize,
    ) -> ffi::Status = ffi::clap_encode_audio_file;
}

#[test]
fn empty_native_output_never_forms_a_null_slice() {
    assert!(NativeBuffer::default().into_vec().unwrap().is_empty());
}

#[test]
fn nonempty_null_native_output_is_rejected() {
    let output = NativeBuffer {
        values: std::ptr::null_mut(),
        count: 1,
    };
    assert!(output.into_vec().unwrap_err().to_string().contains("null"));
}

#[test]
fn status_diagnostic_is_owned_before_another_native_call() {
    let mut out = std::ptr::null_mut();
    let status = unsafe { ffi::clap_create(std::ptr::null(), std::ptr::null(), &mut out) };
    let error = check_status(status).unwrap_err();
    let message = error.to_string();
    assert!(!message.is_empty());
    let mut second = std::ptr::null_mut();
    let missing = CString::new("missing-ffi-regression.gguf").unwrap();
    let second_status =
        unsafe { ffi::clap_create(missing.as_ptr(), std::ptr::null(), &mut second) };
    assert!(check_status(second_status).is_err());
    assert_eq!(error.to_string(), message);
    assert!(out.is_null());
    assert!(second.is_null());
}
