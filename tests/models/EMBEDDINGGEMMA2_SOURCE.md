# EmbeddingGemma 2 text/audio inference contract

Primary sources, pinned for reproducibility:

- llama.cpp `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`,
  `src/models/gemma-embedding2.cpp`, `conversion/gemma.py`, `src/llama-vocab.cpp`.
- Transformers `95bafdcccfffe17ec383999fc39273867b11385e`,
  `src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py`.
- Google `google/embeddinggemma-2`, revision
  `26711154b641697c443124e316f3d21587f9ff5d` (config/tokenizer/pooling).
- Unsloth `unsloth/embeddinggemma-2-GGUF`, revision
  `031f0d4b35536f69ab3509d4893c923264fcf253`.

The text and audio graph contracts were also checked against llama.cpp
`dd6b945be2188da875ab05a259ce94c6b2736e64`. Its text changes are graph/RoPE
helper refactoring; its audio changes add streaming state, outside this
non-streaming encoder's scope. The pinned revision above remains the numerical
reference for reproducibility.

## Scope and numerical details

Load the standard `gemma-embedding2` GGUF directly. Text requires only the main
model; audio and images also require a matching `clip` mmproj with the
corresponding Gemma4 tower. No checkpoint converter is added. Default output is
768D; retrieval prefixes are explicit input text, not automatically inserted.

The published model has 24 blocks, width 512, FFN width 2048, four Q heads,
and 512D per-layer projection inputs. Blocks 5/11/17/23 (zero based) use global
attention with head dimension 512 and one KV head. Local blocks use dimension
256 and two KV heads. Attention is bidirectional, scale 1.0. GGUF window 1024
means inclusive radius 512, not radius 1024. RoPE is half-split NeoX, local
base 10000/global base 1000000. RMS weights multiply directly; do not add 1.
Q/K have learned per-head RMS weights; V RMS has no learned weight. Apply
post-attention/FFN/PLE norms before their residual additions, then learned
layer-output scaling. PLE comes from the initial scaled embeddings, not each
block's hidden states. Final projection precedes masked mean pooling and L2
normalization. Include BOS/EOS/prompt; exclude pad positions.

Gemma4 tokenization is raw-UTF-8 BPE using explicit merge ranks. Replace ASCII
spaces with `▁` without Unicode normalization, whitespace collapsing, or dummy
leading space. Honor newline runs, byte fallback, and special tokens. The
published GGUF adds both BOS=2 and EOS=1. Scores are not BPE merge priorities.

Preserve F32 activations/reductions even when weights use F16/BF16 storage.
The shared storage-only matmul path decodes its current operand; quantized
weights are not a license to round all intermediate activations to F16.

## Default automated tests

`embeddinggemma2.py` generates deterministic valid tiny GGUFs and an independent
stdlib scalar oracle. F32/F16 storage, mixed local/global heads, inclusive local
window boundaries, PLE, padding, strings/IDs, pooling, malformed metadata and
shapes are covered. `embeddinggemma2_cli.py` checks CLI and HTTP vectors against
that oracle. All fixtures are generated outside the source tree. Dedicated
native tests cover tokenizer rules and Q5_K/BF16 storage.

`gemma_tokenizer_parity.py` is an optional published-tokenizer parity generator;
it requires the `tokenizers` package only for testing, never for inference.
It checks the pinned tokenizer's SHA256 before generating cases. Enable its
CTest fixture and native comparison with:

```sh
cmake -S . -B build -DCLAP_GEMMA_TOKENIZER_JSON=/path/to/tokenizer.json
ctest --test-dir build -R test-gemma-tokenizer-hf --output-on-failure
```

## Published checkpoint validation

Validated `embeddinggemma-2-UD-Q4_K_XL.gguf` (175673856 bytes), SHA256
`ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493`.
This file mixes Q4_K/Q5_K/Q6_K/Q8_0/F32 tensors. Six actual inputs were compared
against unmodified pinned llama.cpp: English, Chinese, query/document prefixes,
and a 602-token input crossing the local attention radius.

Use `llama-embedding` with `--pooling mean --embd-normalize 2
--embd-output-format json --no-warmup -c 1024 -b 1024 -ub 1024` and the exact
same literal prompt. Native packed quantized llama.cpp versus this runtime's
F32-activation path had minimum cosine 0.9995059. To separate activation packing
from architecture errors, export the **same quantized weights** losslessly to
F32 storage with the test-only `test-q8 decode INPUT OUTPUT`, then run that
F32 model in llama.cpp. Across the same six inputs, minimum cosine was
0.999999926 and maximum absolute component error was 5.332e-5. This is a
same-stored-weight oracle, not a claim of parity against Google's original
unquantized checkpoint or an embedding-quality benchmark.

## Native audio contract

Additional pinned primary source files are Transformers' Gemma4
`feature_extraction_gemma4.py`, `modeling_gemma4.py` and EmbeddingGemma2's
`processing_embedding_gemma2.py`, plus llama.cpp's
`tools/mtmd/models/gemma4a.cpp` and `tools/mtmd/clip.cpp` at the revisions above.

Audio remains native waveform inference, not ASR-to-text. Decode/downmix/resample
through the existing audio module to 16 kHz. Keep the first 480000 samples
(published default 30-second truncation). Use 128 HTK magnitude-mel bins, a
periodic 320-sample Hann window, hop160, FFT512, semicausal left160 padding and
321-sample frame availability; consume the first320 window samples. Features
are natural `log(mel + 0.001)`, not power, base10 log or Slaney-normalized mel.
The real frame-end mask gives `(samples - 161) // 160 + 1` valid frames;
samples <=160 have no valid token and are rejected. The two stride2 SSCP
convolutions yield `ceil(frames/4)` tokens: one second gives99/25, 30 seconds
gives2999/750. The processor's separate 280-token estimate is not a truncation
limit on actual audio features or soft tokens.

SSCP uses channel LayerNorm/ReLU and frequency-major/channel-minor flattening.
The 12 Conformer blocks use two half-scaled residual SiLU FFNs, clipped linear
inputs/outputs, converted per-dimension query scales (already softplus), strict
causal attention distance `<12`, and causal kernel5 depthwise convolution.
GGUF `conv_norm` is the pre-convolution RMS; `norm_conv` is post-depthwise RMS.
All480 learned clipping scalars are required. The audio tower outputs1536D,
unweighted RMS then `mm.a.input_projection` produces512D soft tokens.

Merge `[BOS, BOA] + soft + [EOA, EOS]`, scaling hard word rows by sqrt512 but
not the projected soft rows. Derive PLE from the merged embeddings and reuse
the same bidirectional text transformer, per-token output projection, mean
over every real position and L2 normalization. Unknown `a.*` and `mm.a.*`
tensors are rejected; unrelated vision tensors in the mixed mmproj are ignored.

### Audio tests and checkpoint smoke validation

`embeddinggemma2_audio.py` is an independent stdlib scalar/DFT oracle for
F32/F16 storage, SSCP order, chunk boundaries, clipping and optional projector
normalization. `embeddinggemma2_audio_integration.py` composes that oracle with
the independent text oracle and exercises public waveform encoding, automatic
resampling and WAV CLI output. Native tests cover frame/token boundaries,
750 tokens at30 seconds, truncation equivalence, malformed tensors, unknown
audio prefixes, nonfinite input and integer-overflow metadata. No model or
audio binaries, NumPy, Torch or Transformers are required by default tests.

Validated the published `mmproj-Q8_0.gguf` (554821120 bytes), SHA256
`90e7b0238009e2954f856f2081dcf7f35af026b64c765e98f4777053e1754460`.
Its 963 tensors include 751 audio tensors and a separate audio projector. On
two distinct one-second PCM16 waveforms (440 Hz sine and chirp), the published
Q4 text checkpoint plus Q8 audio projector produced finite normalized 768D
outputs, with mutual cosine 0.93683089.

For independent architecture comparison, the exact official HF log-mel
features were supplied to the unmodified pinned llama.cpp audio tower and
text encoder, using the same stored quantized weights decoded losslessly to
F32. Against those saved reference vectors, fresh native inference had minimum
final cosine 0.9999996308 and maximum absolute component error 1.35035e-4.
The official frontend had previously matched the native frontend within
4.7684e-7. These are numerical checks, not retrieval-quality measurements.

Two reference limitations must remain explicit: the pinned llama.cpp waveform
frontend computes `log(max(mel, 0.001))`, unlike official/native
`log(mel + 0.001)`, so direct waveform comparisons across those frontends are
not architecture-parity evidence. Its SSCP convolutions also pack inputs to
F16 even with decoded-F32 weights. The identical-feature comparison separates
the frontend discrepancy but does not claim bit-exact inference or attribute
every residual difference to packing. No GPU or original unquantized Google
checkpoint validation is claimed.

## Native image contract

Image support extends the earlier text/audio scope above. The decoder produces
packed top-left RGB8 with stb_image; resizing and the complete vision model are
native C++ in `src/models/embeddinggemma2.cpp`, beside the audio/text models.
The private `GemmaVisionEncoder` declaration lives in `src/model.h`.

Additional primary references are Transformers
`95bafdcccfffe17ec383999fc39273867b11385e`:
`gemma4/image_processing_gemma4.py`, `gemma4/modeling_gemma4.py`, and
`embedding_gemma2/processing_embedding_gemma2.py`; and pinned llama.cpp
`tools/mtmd/models/gemma4v.cpp`/`tools/mtmd/clip.cpp`. The Gemma4v source at
llama.cpp `5d4c8eeb24db049acf74bd6f354bed6eaeffc7dc` differs from the pin only
by the `clip_graph`/`clip_layer` to `mtmd_graph`/`mtmd_layer` type renaming.

The published mmproj has patch16, hidden768, 16 blocks, 12 heads of width64,
FFN3072, pooling kernel3, and a learned `[768,10240,2]` X/Y position table.
Use the official processor's default budget of 280 soft tokens (2520 patches),
not the mmproj's nominal `image_size=224` as a forced square resize. An optional
`clip.vision.image_max_pixels` overrides the budget; this also permits tiny
self-contained test fixtures. Resize preserves aspect ratio, rounds each side
down to a patch*pool multiple, and handles extreme-aspect one-block axes.
The antialiased Keys bicubic uint8 path uses per-axis int16 coefficient
quantization, uint8 horizontal intermediates and half-up output rounding,
matching torchvision/ATen, not an arbitrary bicubic image-resizer default.

Flatten patches in GGUF convolution order (channel, Y, X), apply `2*RGB/255-1`,
then the bias-free patch projection and both learned position lookups. Every
block uses weighted pre/post attention and pre/post FFN RMS, weighted per-head
Q/K RMS, unweighted per-head V RMS, independent axial NeoX RoPE with theta100,
unit attention scale and gated tanh-GELU. Padded patch keys are masked in HF;
native inference drops all padded rows, which preserves real-query attention
and pooling exactly. Average spatial3x3 groups and multiply by sqrt768, apply
optional stored standardization, unweighted RMS and `mm.input_projection`.
Optional learned linear input/output clipping is honored. Unknown `v.*` and
`mm.*` tensors are rejected, excluding the independently validated `mm.a.*`.
Disabled modalities may retain their metadata without requiring their tensors.

Compose `[BOS, <|image>] + projected soft rows + [<image|>, EOS]`, scaling only
hard vocabulary rows by sqrt(text hidden). PLE, bidirectional transformer,
per-token final projection and mean/L2 reuse the exact text/audio backbone.
Missing or disabled vision towers reject image inference independently of audio.

`embeddinggemma2_image.py` and `test-embeddinggemma2-image.cpp` generate valid
F32/F16 tiny GGUFs and compare 20 image/model variants against independent
stdlib scalar vision and text oracles. Cases cover unchanged/up/downsampled,
square, landscape, portrait and extreme-aspect RGB; nontrivial learned X/Y
positions and multiple axial frequencies; optional standardization/clipping;
packed RGB and PNG-file public APIs; malformed geometry/flags, unknown tower
and projector tensors; and disabled-but-described towers. Default tests need
neither downloads nor Torch/Pillow/NumPy. Optional second CLI argument checks
`clap embedding --image` against the same scalar embeddings.

In isolated static-build execution, all 80 stage/API numerical comparisons
passed with maximum absolute error `4.91738e-6` (tolerance `1e-5`); original
text and audio scalar suites remained green. Separately, the stdlib resize
reference matched actual torchvision uint8 antialiased bicubic byte-for-byte
on seven upsample/downsample/aspect cases, including 96x96 to768x768 and
157x83 to1056x528. These checks do not by themselves establish published-model
end-to-end parity or retrieval quality.
