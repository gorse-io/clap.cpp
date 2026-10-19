# Native CLAP model

`clap-models` implements the unfused HuggingFace `laion/larger_clap_music`
checkpoint in C++17. `clap-model` is the real inference executable. The original
`clap embedding` command and HTTP server intentionally remain dummy interfaces;
they are not connected to this model and retain their existing tests.

## Build

Install ICU development headers/libraries (`libicu-dev` on Debian/Ubuntu), then:

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Inference requires no Python, PyTorch, network connection, or tokenizer sidecar.
Dense operations and lowered patch convolution use ggml's CPU backend; attention,
normalization, preprocessing, and tokenization are native C++. GPU execution is
not implemented even if other ggml backends are enabled.

## Convert and run

Conversion alone requires Python with `torch`, `transformers`, and `numpy`.
Download `config.json`, `pytorch_model.bin`, `tokenizer.json`, and
`preprocessor_config.json` from `laion/larger_clap_music` into a local directory.
The tested HuggingFace revision is `a0b4534a14f58e20944452dff00a22a06ce629d1`.
Only convert trusted checkpoints; the converter uses `torch.load` with
`weights_only=True` and memory mapping, never unrestricted pickle loading.

```sh
python src/models/convert_clap.py checkpoint/ larger-clap-music.gguf
# Optional smaller storage; F16 matrix kernels also round their inputs:
python src/models/convert_clap.py checkpoint/ larger-clap-music-f16.gguf --dtype f16

./build/clap-model --model larger-clap-music.gguf --text 'A jazz quartet playing saxophone.'
./build/clap-model --model larger-clap-music.gguf --audio music.wav --threads 2
```

Output is one JSON array containing 512 normalized floating-point values. WAV
input must be mono 48 kHz PCM16 or IEEE float32. Resample/downmix other audio
externally. Audio longer than ten seconds uses its first ten seconds by default;
`--crop-offset N` selects another crop by sample offset. This is an explicit,
reproducible replacement for HuggingFace's random `rand_trunc` selection, not an
attempt to reproduce NumPy's RNG. Short input uses HF `repeatpad`: repeat complete
copies, then zero-pad the remainder. Empty/non-finite input is rejected.

## C++ API

Link the CMake target `clap-models`, include `models/clap_model.h`, construct
`clap::Model` with a GGUF filename, and call `text(utf8)` or `audio(pcm, 48000)`.
`tokenize` returns RoBERTa byte-level BPE IDs including BOS/EOS, truncating to 512
tokens by default. Unicode pre-tokenization uses ICU. `text(token_ids)` accepts
already-tokenized input and masks padding ID 1. `audio_features(features, frames)`
accepts a flattened, single-channel time-major log-mel spectrogram. `log_mel`
exposes the waveform frontend separately.

Model instances are move-only and are not safe for concurrent calls. POSIX
weights are mapped read-only; Windows uses a buffered fallback (not tested).
The GGUF contains the fully expanded HF config and tokenizer as string metadata
`clap.config` and `clap.tokenizer`, with `general.architecture=clap`. Tensor names
retain HF structure with shortened model/projection prefixes and `rel_bias`.
Tensor dimensions reverse the PyTorch shape as required by ggml; data stay in
contiguous PyTorch order. F32 and F16 storage are supported, not quantized types.
The loader validates required tensor shapes and rejects truncated tensor data.
Fusion-enabled CLAP, other patch geometry, and non-GELU encoders/non-ReLU
projections are rejected rather than silently treated as this architecture.

## Architecture and preprocessing

The audio branch includes mel-bin batch normalization, bicubic interpolation
(`align_corners=True`, cubic coefficient -0.75), HTSAT mel-to-image folding,
4x4 patch convolution, patch layer norm, four Swin stages with depths
`[2,2,12,2]`, shifted-window masks and relative-position biases, patch merging,
final layer norm and global average pooling. The text branch includes RoBERTa
word/position/type embeddings, post-norm Transformer layers with padding masks,
and the tanh CLS pooler. Both branches use their learned two-layer ReLU
projection and L2 normalization.

The waveform frontend uses 48 kHz, a periodic Hann window of length 1024,
centered reflect-padded STFT with hop 480, power spectra, 64 Slaney-normalized
Slaney mel filters from 50 to 14000 Hz, and power-to-dB with a 1e-10 floor and no
top-dB clipping. Ten seconds yields 1001 frames, despite `nb_max_frames=1000` in
the processor metadata.

Reference sources: the checkpoint's HF config/preprocessor/tokenizer and
Transformers `models/clap/modeling_clap.py`, `feature_extraction_clap.py`, and
`audio_utils.py`. The original implementation was researched against
Transformers 4.44.2 source; numerical tests ran against installed Transformers
5.13.1 and CPU PyTorch 2.14.1.

## Tests

The default CTest suite needs no model download or ML packages. Optional oracle
tests construct a deterministic small real Transformers CLAP (not mocked),
convert it, and compare native text/audio embeddings, Unicode BPE, padding,
shifted windows, patch merging, projections, and preprocessing. They also check
F16 storage, malformed GGUF files, unsupported fusion config, and CLI behavior.

```sh
cmake -S . -B build -DCLAP_MODEL_PARITY=ON -DPython3_EXECUTABLE=/path/to/python
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure

# Actual pretrained checkpoint oracle; does not download anything:
python tests/models/checkpoint_parity.py checkpoint/ build/ parity-output/
```

Actual checkpoint F32 parity was exercised for four text prompts (including
Unicode and empty text), synthetic chirp audio, and native waveform inference.
Observed maximum absolute error was below 1e-7 for embeddings and 7.63e-6 for
log-mel features. The public native CLI also produced matching 512-dimensional
text/audio embeddings. Tiny-model F16 errors were below 1.8e-4; its oracle uses
a 1e-3 tolerance because the ggml F16 dot path also rounds activations. Full
checkpoint F16 accuracy, Windows/macOS, GPU execution, real-world audio retrieval
quality, HTTP inference, batching, and quantization have not been verified.
