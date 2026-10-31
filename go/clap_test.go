package clap

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
)

func fixture(name string) string { return filepath.Join("build", "fixtures", name) }
func openModel(t *testing.T, name string) *CLAP {
	t.Helper()
	m, err := New(fixture(name), 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}
func near(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("dimensions: %d != %d", len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v-want[i])) > tolerance {
			t.Fatalf("embedding[%d]: %g != %g", i, v, want[i])
		}
	}
}
func TestTextReferenceAndOwnership(t *testing.T) {
	m := openModel(t, "模型.gguf")
	b, err := os.ReadFile(fixture("text-reference.txt"))
	if err != nil {
		t.Fatal(err)
	}
	var want []float32
	for _, s := range strings.Fields(string(b)) {
		v, e := strconv.ParseFloat(s, 32)
		if e != nil {
			t.Fatal(e)
		}
		want = append(want, float32(v))
	}
	got, err := m.EncodeText("Hi")
	if err != nil {
		t.Fatal(err)
	}
	near(t, got, want, 2e-5)
	got[0] = 999
	again, err := m.EncodeText("Hi")
	if err != nil {
		t.Fatal(err)
	}
	near(t, again, want, 2e-5)
	if _, err = m.EncodeText("音乐 🎵"); err != nil {
		t.Fatal(err)
	}
}
func TestAudioDownmixResampleCrop(t *testing.T) {
	m := openModel(t, "模型.gguf")
	mono, err := m.EncodeAudio(fixture("音频.wav"), 0)
	if err != nil {
		t.Fatal(err)
	}
	stereo, err := m.EncodeAudio(fixture("立体声.wav"), 0)
	if err != nil {
		t.Fatal(err)
	}
	near(t, stereo, mono, 2e-5)
	cropped, err := m.EncodeAudio(fixture("long.wav"), 48000)
	if err != nil {
		t.Fatal(err)
	}
	control, err := m.EncodeAudio(fixture("crop.wav"), 0)
	if err != nil {
		t.Fatal(err)
	}
	near(t, cropped, control, 2e-5)
	mert := openModel(t, "mert.gguf")
	v, err := mert.EncodeAudio(fixture("mert.wav"), 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 4 {
		t.Fatalf("MERT dimension %d", len(v))
	}
	if _, err = mert.EncodeText("Hi"); err == nil {
		t.Fatal("audio-only model accepted text")
	}
}
func TestErrors(t *testing.T) {
	for _, p := range []string{"missing.gguf", fixture("invalid.wav"), "bad\x00path", string([]byte{255})} {
		if m, e := New(p, 1); e == nil {
			m.Close()
			t.Errorf("accepted model %q", p)
		}
	}
	if m, e := New(fixture("模型.gguf"), -1); e == nil {
		m.Close()
		t.Fatal("accepted negative threads")
	}
	if m, e := New(fixture("模型.gguf"), int(^uint(0)>>1)); e == nil {
		m.Close()
		t.Fatal("accepted overflowing threads")
	}
	m := openModel(t, "模型.gguf")
	for _, s := range []string{"bad\x00text", string([]byte{255})} {
		if _, e := m.EncodeText(s); e == nil {
			t.Errorf("accepted text %q", s)
		}
	}
	for _, p := range []string{"missing.wav", fixture("invalid.wav"), "bad\x00path", string([]byte{255})} {
		if _, e := m.EncodeAudio(p, 0); e == nil {
			t.Errorf("accepted audio %q", p)
		}
	}
	if _, e := m.EncodeAudio(fixture("long.wav"), ^uint64(0)); e == nil {
		t.Fatal("accepted crop overflow")
	}
}
func TestCloseAndCopies(t *testing.T) {
	var zero CLAP
	var nilModel *CLAP
	for _, m := range []*CLAP{&zero, nilModel} {
		if e := m.Close(); e != nil {
			t.Fatal(e)
		}
		if _, e := m.EncodeText("Hi"); !errors.Is(e, ErrClosed) {
			t.Fatalf("zero text error %v", e)
		}
		if _, e := m.EncodeAudio("x", 0); !errors.Is(e, ErrClosed) {
			t.Fatalf("zero audio error %v", e)
		}
	}
	m := openModel(t, "模型.gguf")
	copy := *m
	if e := copy.Close(); e != nil {
		t.Fatal(e)
	}
	if e := m.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e := m.EncodeText("Hi"); !errors.Is(e, ErrClosed) {
		t.Fatalf("copy close: %v", e)
	}
}
func TestConcurrentInferenceAndClose(t *testing.T) {
	m := openModel(t, "模型.gguf")
	copy := *m
	want, err := m.EncodeText("Hi")
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	// First exercise actual concurrent inference before allowing Close to win.
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := copy.EncodeText("Hi")
			if err != nil {
				t.Error(err)
				return
			}
			near(t, got, want, 2e-5)
		}()
	}
	wg.Wait()
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 8; j++ {
				_, e := copy.EncodeText("Hi")
				if e != nil && !errors.Is(e, ErrClosed) {
					t.Error(e)
				}
			}
		}()
	}
	wg.Add(1)
	go func() { defer wg.Done(); m.Close() }()
	wg.Wait()
}
