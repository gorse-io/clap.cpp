// embed runs both modalities using a locally installed clap-go native bridge.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	clap "github.com/gorse-io/clap.cpp/go"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "usage: embed model.gguf text audio.wav")
		os.Exit(2)
	}
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	model, err := clap.New(os.Args[1], 1)
	if err != nil {
		return err
	}
	defer model.Close()
	text, err := model.EncodeText(os.Args[2])
	if err != nil {
		return err
	}
	audio, err := model.EncodeAudio(os.Args[3], 0)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(struct {
		Text  []float32 `json:"text"`
		Audio []float32 `json:"audio"`
	}{text, audio})
}
