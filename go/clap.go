// Package clap provides native text and audio embeddings through the clap C API.
package clap

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo windows CFLAGS: -DCLAP_SHARED
#cgo LDFLAGS: -lclap
#include <clap.h>
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unicode/utf8"
	"unsafe"
)

// ErrClosed is returned when inference is attempted on a closed or zero handle.
var ErrClosed = errors.New("clap: model is closed")

// CLAP owns a native model. Its zero value is closed. Copies share ownership:
// closing any copy closes all copies. Calls (including Close) are serialized.
type CLAP struct{ state *modelState }
type modelState struct {
	mu    sync.Mutex
	model *C.CLAP
}

func validString(s string) error {
	if !utf8.ValidString(s) || strings.IndexByte(s, 0) >= 0 {
		return errors.New("clap: strings must be valid UTF-8 without NUL")
	}
	return nil
}

// The caller must stay on the same OS thread between a fallible C API call
// and reading its thread-local diagnostic.
func statusError(status C.clap_status) error {
	if status != C.CLAP_OK {
		return fmt.Errorf("clap: %s", C.GoString(C.clap_last_error()))
	}
	return nil
}

// New loads a GGUF model. threads=0 selects the native default; negative values
// and values outside the native 32-bit thread range are rejected.
func New(modelPath string, threads int) (*CLAP, error) {
	if err := validString(modelPath); err != nil {
		return nil, err
	}
	if threads < 0 || uint64(threads) > 1<<31-1 {
		return nil, errors.New("clap: threads out of range")
	}
	path := C.CString(modelPath)
	defer C.free(unsafe.Pointer(path))
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	options := C.clap_default_options()
	options.threads = C.int(threads)
	var model *C.CLAP
	if err := statusError(C.clap_create(path, &options, &model)); err != nil {
		return nil, err
	}
	s := &modelState{model: model}
	runtime.SetFinalizer(s, func(s *modelState) { s.close() })
	return &CLAP{state: s}, nil
}

func (s *modelState) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.model != nil {
		C.clap_destroy(s.model)
		s.model = nil
	}
}

// Close releases the model. It is safe to call repeatedly, including on nil.
// Explicit Close is preferred; a finalizer is only a fallback.
func (m *CLAP) Close() error {
	if m != nil && m.state != nil {
		m.state.close()
		runtime.KeepAlive(m.state)
	}
	return nil
}

func (m *CLAP) encode(input string, audio bool, crop uint64) ([]float32, error) {
	if m == nil || m.state == nil {
		return nil, ErrClosed
	}
	s := m.state
	s.mu.Lock()
	defer s.mu.Unlock()
	defer runtime.KeepAlive(s)
	if s.model == nil {
		return nil, ErrClosed
	}
	if err := validString(input); err != nil {
		return nil, err
	}
	// Native preprocessing uses signed indices as well as size_t.
	if crop > uint64(^uint(0)>>1) {
		return nil, errors.New("clap: crop offset out of range")
	}
	dimension := int(C.clap_projection_dimension(s.model))
	if dimension <= 0 || uint64(dimension) > uint64(^uint(0)>>1)/4 {
		return nil, errors.New("clap: invalid embedding dimension")
	}
	out := make([]float32, dimension)
	str := C.CString(input)
	defer C.free(unsafe.Pointer(str))
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var status C.clap_status
	values := (*C.float)(unsafe.Pointer(&out[0]))
	if audio {
		status = C.clap_encode_audio_file(s.model, str, C.size_t(crop), values, C.size_t(len(out)))
	} else {
		status = C.clap_encode_text(s.model, str, values, C.size_t(len(out)))
	}
	if err := statusError(status); err != nil {
		return nil, err
	}
	return out, nil
}

// EncodeText embeds a UTF-8 string. Returned storage belongs to Go.
func (m *CLAP) EncodeText(text string) ([]float32, error) { return m.encode(text, false, 0) }

// EncodeAudio decodes WAV/MP3/FLAC, downmixes and resamples for the model.
// cropOffset is in samples at the model's rate, not bytes or source frames;
// its interpretation/support is architecture-specific (CLAP: 10-second crop).
func (m *CLAP) EncodeAudio(path string, cropOffset uint64) ([]float32, error) {
	return m.encode(path, true, cropOffset)
}
