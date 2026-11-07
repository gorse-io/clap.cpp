// Package clap provides native CPU text and audio embeddings through CGO.
// Build and install go/native first; see README.md for pkg-config/loader setup.
package clap

/*
#cgo pkg-config: clap-go
#include <stdlib.h>
#include <clap-go/bridge.h>
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
	model *C.cg_model
}

func validString(s string) error {
	if !utf8.ValidString(s) || strings.IndexByte(s, 0) >= 0 {
		return errors.New("clap: strings must be valid UTF-8 without NUL")
	}
	return nil
}
func resultError(r *C.cg_result) error {
	if r == nil {
		return errors.New("clap: native result allocation failed")
	}
	if s := C.GoString(C.cg_error(r)); s != "" {
		return fmt.Errorf("clap: %s", s)
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
	r := C.cg_open(path, C.int(threads))
	if r != nil {
		defer C.cg_result_free(r)
	}
	if err := resultError(r); err != nil {
		return nil, err
	}
	s := &modelState{model: C.cg_take_model(r)}
	runtime.SetFinalizer(s, func(s *modelState) { s.close() })
	return &CLAP{state: s}, nil
}
func (s *modelState) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.model != nil {
		C.cg_model_free(s.model)
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
	// Native preprocessing uses signed indices as well as size_t. Reject values
	// outside either range before conversion, including on 32-bit targets.
	if crop > uint64(^uint(0)>>1) {
		return nil, errors.New("clap: crop offset out of range")
	}
	str := C.CString(input)
	defer C.free(unsafe.Pointer(str))
	var r *C.cg_result
	if audio {
		r = C.cg_audio(s.model, str, C.size_t(crop))
	} else {
		r = C.cg_text(s.model, str)
	}
	if r != nil {
		defer C.cg_result_free(r)
	}
	if err := resultError(r); err != nil {
		return nil, err
	}
	n := uint64(C.cg_size(r))
	if n > uint64(^uint(0)>>1)/4 {
		return nil, errors.New("clap: embedding too large")
	}
	out := make([]float32, int(n))
	if n > 0 {
		copy(out, unsafe.Slice((*float32)(unsafe.Pointer(C.cg_data(r))), int(n)))
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
