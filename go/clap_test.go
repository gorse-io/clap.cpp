package clap

import (
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func cosine(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		normA += x * x
		normB += y * y
	}
	return dot / math.Sqrt(normA*normB)
}

func TestCLAP(t *testing.T) {
	modelPath := filepath.Join("..", "models", "larger_clap_general-q4_k_m.gguf")
	info, err := os.Stat(modelPath)
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	if os.IsNotExist(err) || info.Size() == 0 {
		if err := os.MkdirAll(filepath.Dir(modelPath), 0755); err != nil {
			t.Fatal(err)
		}
		client := http.Client{Timeout: 10 * time.Minute}
		response, err := client.Get("https://huggingface.co/gorse-io/larger_clap_general-GGUF/resolve/main/larger_clap_general-q4_k_m.gguf")
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			t.Fatalf("model download: %s", response.Status)
		}
		partial := modelPath + ".part"
		defer os.Remove(partial)
		file, err := os.Create(partial)
		if err != nil {
			t.Fatal(err)
		}
		_, copyErr := io.Copy(file, response.Body)
		closeErr := file.Close()
		if copyErr != nil {
			t.Fatal(copyErr)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		if err := os.Rename(partial, modelPath); err != nil {
			t.Fatal(err)
		}
	}

	model, err := New(modelPath, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	audio, err := model.EncodeAudio(filepath.Join("..", "assets", "dog.wav"), 0)
	if err != nil {
		t.Fatal(err)
	}
	text1, err := model.EncodeText("Sound of a dog")
	if err != nil {
		t.Fatal(err)
	}
	text2, err := model.EncodeText("Sound of vaccum cleaner")
	if err != nil {
		t.Fatal(err)
	}
	if len(audio) == 0 || len(audio) != len(text1) || len(audio) != len(text2) {
		t.Fatalf("embedding dimensions: %d, %d, %d", len(audio), len(text1), len(text2))
	}
	// Learned logit_scale_a from laion/larger_clap_general, as in the C++ test.
	logitScale := math.Exp(3.6549275)
	logit1, logit2 := cosine(audio, text1)*logitScale, cosine(audio, text2)*logitScale
	maxLogit := math.Max(logit1, logit2)
	exp1, exp2 := math.Exp(logit1-maxLogit), math.Exp(logit2-maxLogit)
	score1, score2 := exp1/(exp1+exp2), exp2/(exp1+exp2)
	if !(score1 > 0.99 && score2 < 0.01) {
		t.Fatalf("similarity scores: dog=%g, vacuum cleaner=%g", score1, score2)
	}
	t.Logf("similarity scores: dog=%g, vacuum cleaner=%g", score1, score2)
}
