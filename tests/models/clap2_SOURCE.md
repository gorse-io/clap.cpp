# CLAP2 independent test provenance

## Reviewed upstream implementations

The numerical oracle executes unmodified `mlfoundations/open_clip` source at
`f9a0948d573009f8fa3e530e55814cb11fb8e8bd`, plus **timm 1.0.30**. It does not
reimplement a second transformer that could repeat native inference mistakes.
Relevant upstream files:

- `src/open_clip/model_configs/naflexclap_large.json`
- `src/open_clip/audio/naflex_tower.py`: timm axial-RoPE audio transformer
- `src/open_clip/audio/naflex_audio.py`: sub-window waveform padding, uncapped
  variable-length patchification, frequency-major/time-minor patch ordering
- `src/open_clip/audio/dsp.py`: centered reflect-padded power HTK log-mel frontend
- `src/open_clip/transformer.py`: ModernTextTransformer and masked MAP pool
- `src/open_clip/tokenizer.py`: TikTokenTokenizer, `encode_ordinary` semantics

Published HF assets/checkpoint are `laion/clap2-large-audioxl-s1B` revision
`e836feaa79cd1f7de939fe77065819f48fc44e4c`. The fixture generator validates its
complete reduced-depth HF tensor-name set against the downloaded checkpoint
header. Production conversion keeps those HF tensor names unchanged.

### Important text MLP layout difference

Official open_clip **text** SwiGLU `w12` stores `[value, gate]`; HF
`gate_up_proj` stores `[gate, value]`. `hf_value` swaps these halves when creating
an HF fixture or importing a published HF checkpoint into the official oracle.
Audio timm SwiGLU is already gate-first. This bridge is test-only; the production
converter must NOT swap a checkpoint that is already in HF layout.

The distinction was checked against pinned published checkpoint bytes, not just
inferred from names: an HTTP range read of the first 128 bytes of open_clip
`text.blocks.0.mlp.w12.weight` matches the HF tensor's **second** half, not its
first. The open_clip header is 27,992 bytes (data starts at 28,000), and this
weight starts at relative data offset 680,594,436. Separate range-read controls
for text/audio block-0 QKV, audio block-0 MLP fc1 and text pool KV are byte-identical
to their corresponding HF tensors.

## Native runner

Build `clap2-model-test` from `tests/models/clap2-model-test.cpp`, linked with
`clap` and `nlohmann_json::nlohmann_json`. It uses the public C API and existing
`tests/c-api-helpers.h`; no private include directories are needed.

```
"$BUILD/clap2-model-test" MODEL.gguf FIXTURE.json
```

It forces CPU, two threads and F32 activation compute. F16 below means rounded
weight storage, not half-precision activations. PCM fixtures are generated
little-endian IEEE F32; the runner handles host byte order.

## Default stdlib-only regression

```
python tests/models/clap2_invalid.py "$BUILD" "$BUILD/clap2-smoke"
```

This generates complete analytical F32/F16 GGUFs, checks tokenizer/text/token/audio
public APIs, checks converter metadata/schema, and rejects unsupported native
configuration, tokenizer templates, missing tensors and wrong tensor shapes.
It requires no model downloads or optional Python ML packages. Metadata discovery
is tested independently of safetensors payload conversion; actual F32/F16
conversion is covered by the optional numerical oracle below.

`clap2_fixture.py` can also generate its analytical smoke fixture separately:

```
python tests/models/clap2_fixture.py "$BUILD/clap2-smoke"
"$BUILD/clap2-model-test" "$BUILD/clap2-smoke/model-f32.gguf" "$BUILD/clap2-smoke/fixture.json"
"$BUILD/clap2-model-test" "$BUILD/clap2-smoke/model-f16.gguf" "$BUILD/clap2-smoke/fixture.json"
```

The degenerate analytical model has constant unit outputs. It establishes valid
loader/API smoke coverage, NOT numerical architecture/frontend parity.

## Optional official tiny numerical oracle

```
"$PYTHON" tests/models/clap2_parity.py "$BUILD" "$OUTPUT" "$OFFICIAL_OPEN_CLIP" "$HF_ASSETS"
```

`--generate-only` prepares both GGUFs and fixture JSONs without native execution.
`OFFICIAL_OPEN_CLIP` contains the reviewed checkout, `src/open_clip`, and
`PINNED_SHA` containing the exact source SHA above. `HF_ASSETS` contains pinned
`config.json`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`,
and `model.safetensors.header.json`; no full weights are needed. `OUTPUT` must
be separate from downloaded assets and checkpoints. It is updated in place,
not recursively deleted. tiktoken may fetch/cache its standard r50k_base ranks
on first use; no model checkpoint is downloaded by these scripts.

Exercised dependencies: Python 3.11, torch 2.6.0+cpu, torchvision 0.21.0+cpu,
timm **1.0.30** (enforced), tiktoken 0.14.0, safetensors 0.8.0,
tokenizers 0.21.4, numpy 2.4.6; also open_clip requirements regex, ftfy, tqdm,
huggingface_hub, Pillow and wcwidth. The pinned DSP uses plain torch; torchaudio
and the Python gguf package are not required by this oracle.

Coverage:

- Full two-layer/two-head official audio/text towers with nontrivial learned
  biases, norms, pool probes, registers and layer scales.
- Realistic 32-kHz FFT1024/hop320/mel128/16x16 waveform frontend; sub-window,
  subpatch, nonmultiple, exact multiple, silence, and uncapped 31-second audio.
- An altered tail verifies that the long fixture actually distinguishes a
  silent 30-second crop from complete-waveform inference.
- Exact tiktoken GPT2 byte-BPE contractions/case, Unicode letters/numbers,
  whitespace lookahead, CRLF/tabs, truncation and literal special-token markup.
- Public tokenization returns variable-length IDs; fixed-batch padding is not
  counted as user tokens. The oracle verifies compact/padded inference equality
  and independently tests interior/suffix padding and masked MAP pooling.
- Upstream invalid audio-patch padding invariance.
- F32/F16 storage-rounded parity under a 5e-5 maximum-absolute embedding tolerance.

Tiny random-weight parity is not published-checkpoint parity.

## Optional published checkpoint, one tower per process

```
"$PYTHON" tests/models/clap2_checkpoint_parity.py text "$CHECKPOINT_DIR" "$OUTPUT" "$OFFICIAL_OPEN_CLIP" "$HF_ASSETS" --dtype f32
"$PYTHON" tests/models/clap2_checkpoint_parity.py audio "$CHECKPOINT_DIR" "$OUTPUT" "$OFFICIAL_OPEN_CLIP" "$HF_ASSETS" --dtype f32
"$BUILD/clap2-model-test" MODEL-F32.gguf "$OUTPUT/fixture-text-f32.json"
"$BUILD/clap2-model-test" MODEL-F32.gguf "$OUTPUT/fixture-audio-f32.json"
```

`CHECKPOINT_DIR/model.safetensors` must already exist. Use `--dtype f16` with an
F16-converted GGUF; resulting fixture names end in `-f16.json`. Run native only
after the Python oracle process exits to avoid simultaneous model residency.
The helper constructs only the selected official tower and imports mmap-backed
weights with `assign=True`. Text gate/value halves are restored to open_clip
layout. Audio exercises 600 samples, 5,137 samples and one second; tiny tests
cover long-duration and exhaustive tokenizer regressions separately.

Full checkpoint verification can be disk-I/O-bound on constrained hosts. A
successfully generated reference fixture is not itself native parity evidence;
run both explicit native commands before claiming published-checkpoint parity.
