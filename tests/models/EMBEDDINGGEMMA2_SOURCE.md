# EmbeddingGemma 2 text inference contract

Primary sources, pinned for reproducibility:

- llama.cpp `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`,
  `src/models/gemma-embedding2.cpp`, `conversion/gemma.py`, `src/llama-vocab.cpp`.
- Transformers `95bafdcccfffe17ec383999fc39273867b11385e`,
  `src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py`.
- Google `google/embeddinggemma-2`, revision
  `26711154b641697c443124e316f3d21587f9ff5d` (config/tokenizer/pooling).
- Unsloth `unsloth/embeddinggemma-2-GGUF`, revision
  `031f0d4b35536f69ab3509d4893c923264fcf253`.

## Scope and numerical details

Load the standard `gemma-embedding2` GGUF directly. No checkpoint converter,
image/mmproj, image API, or audio encoder is added. Default output is 768D;
retrieval prefixes are explicit input text, not automatically inserted.

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
ctest --test-dir build -R gemma_tokenizer_hf --output-on-failure
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
F32 storage with the test-only `clap-q8-test decode INPUT OUTPUT`, then run that
F32 model in llama.cpp. Across the same six inputs, minimum cosine was
0.999999926 and maximum absolute component error was 5.332e-5. This is a
same-stored-weight oracle, not a claim of parity against Google's original
unquantized checkpoint or an embedding-quality benchmark.
