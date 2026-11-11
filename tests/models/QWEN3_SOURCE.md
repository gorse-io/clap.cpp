Qwen3 Embedding GGUF source contract
===================================

The runtime reads standard llama.cpp Qwen3 GGUF directly. This project does not
convert or quantize Qwen3 models, does not require embedded Hugging Face
configuration/tokenizer JSON, and rejects the former native-format metadata.
Existing converters and quantizers for other architectures are unchanged.

Immutable references:

- llama.cpp `202c1a229f5a83d1c9d758ed0f2d43760fa4bab0`:
  `src/models/qwen3.cpp`, `src/llama-model.cpp`, `src/llama-vocab.cpp`,
  `gguf-py/gguf/constants.py`, and `conversion/qwen.py`/`conversion/base.py`.
  Source: https://github.com/ggml-org/llama.cpp/tree/202c1a229f5a83d1c9d758ed0f2d43760fa4bab0
- Official `Qwen/Qwen3-Embedding-0.6B-GGUF`, revision
  `370f27d7550e0def9b39c1f16d3fbaa13aa67728`:
  `Qwen3-Embedding-0.6B-f16.gguf`, 1197629632 bytes,
  SHA-256 `421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340`.
  Source: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/tree/370f27d7550e0def9b39c1f16d3fbaa13aa67728

- Official `Qwen/Qwen3-Embedding-4B` config revision
  `5cf2132abc99cad020ac570b19d031efec650f2b` and
  `Qwen/Qwen3-Embedding-8B` config revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.
- Official `Qwen/Qwen3-Embedding-4B-GGUF`, revision
  `f4602530db1d980e16da9d7d3a70294cf5c190be`:
  `Qwen3-Embedding-4B-Q4_K_M.gguf`, 2496703776 bytes,
  SHA-256 `2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b`.
- Official `Qwen/Qwen3-Embedding-8B-GGUF`, revision
  `69d0e58a13e463cd99a9b83e3f5fee7c10265fab`:
  `Qwen3-Embedding-8B-Q4_K_M.gguf`, 4676804928 bytes,
  SHA-256 `3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85`.

Metadata and tensors
--------------------

The model uses typed `qwen3.*` hyperparameters and `tokenizer.ggml.*` arrays.
Integer dimensions/IDs are UINT32, RMS epsilon/RoPE base are FLOAT32, tokenizer
flags are BOOL, tokens/merges are STRING arrays and token types are INT32 arrays.
Tensor names are `token_embd.weight`, `output_norm.weight`, and `blk.N.*`
(`attn_norm`, `attn_q/k/v`, `attn_output`, `attn_q/k_norm`, `ffn_norm`,
`ffn_gate/up/down`). An optional correctly shaped `output.weight` is accepted
but unused for embeddings. Unknown, missing and incorrectly shaped tensors fail.

Published family layouts (all use head dimension 128, vocabulary metadata from
GGUF, RoPE base 1e6 and RMS epsilon 1e-6):

| Variant | Hidden/output | FFN | Blocks | Query/KV heads | Context |
| --- | ---: | ---: | ---: | --- | ---: |
| 0.6B | 1024 | 3072 | 28 | 16/8 | 32768 |
| 4B | 2560 | 9728 | 36 | 32/8 | 40960 |
| 8B | 4096 | 12288 | 36 | 32/8 | 40960 |

The existing model implementation derives every dimension from GGUF; no size
allowlist or production special case is required for 4B/8B. In particular, 4B's
query width 4096 differs from its hidden width 2560, while 8B's widths coincide.
The published HF 4B/8B configs have vocabulary 151665; the GGUF tokenizer and
embedding tensor remain authoritative. An untied optional output head is not
used for embedding inference. Parameter sizes do not select Matryoshka
truncation: output width is the complete model hidden width.

The runtime retains causal contiguous GQA, weighted per-head Q/K RMSNorm,
half-split RoPE with base 1e6, pre-RMSNorm residual blocks, SiLU gated FFN, final RMSNorm and last-token/L2
pooling. Unsupported pooling, noncausal/sliding-window attention and scaled
RoPE are rejected. The existing generic storage reader accepts its supported
F32/F16/BF16/Q8_0/Q4_K/Q6_K types; that is not Qwen3 quantization functionality.

Tokenization follows llama.cpp's `gpt2`/`qwen2` pre-tokenizer and GGUF token types,
merges, IDs and add-BOS/add-EOS flags. In particular it does **not** apply the
NFC normalizer from some HF tokenizer exports. The official GGUF appends EOS
151643 and does not prepend BOS. No chat template or instruction is inserted.
Its padding ID equals EOS; callers of the token API must supply unpadded input,
since an EOS token cannot be distinguished from external batch padding by ID.
Distinct padding IDs are compacted before attention and last-nonpadding pooling.

Verification
------------

Default CTest generates F32/F16 GGUF fixtures in the build directory using only
Python's standard library. An independent scalar decoder covers two nonzero
blocks plus tiny-width 28/36/36-block layouts preserving the 0.6B/4B/8B
query-to-hidden and FFN-to-hidden ratios and GQA grouping. It covers tied and
untied optional output heads, asymmetric head/hidden dimensions, causal
masking, Q/K RMSNorm, RoPE and pooling. It also covers Unicode without NFC, literal special tokens,
BOS/EOS flag combinations, malformed typed metadata, tensor coverage, API/CLI
embeddings and unsupported audio/frame/quantization requests. No weights,
downloads, torch or transformers are required by the default suite.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j1
ctest --test-dir build --output-on-failure
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCLAP_USE_SYSTEM_ICU=ON
cmake --build build-shared -j1
ctest --test-dir build-shared --output-on-failure
```

For independent published-GGUF verification, build the pinned external
llama.cpp as a shared library and compile the optional reference helper:

```sh
cmake -S "$LLAMA_SOURCE" -B "$LLAMA_BUILD" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build "$LLAMA_BUILD" --target llama -j1
c++ -std=c++17 tests/models/qwen3-llama-reference.cpp \
  -I"$LLAMA_SOURCE/include" -I"$LLAMA_SOURCE/ggml/include" -Ithirdparty/json/include \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" -lllama -lggml -lggml-cpu -lggml-base -o "$LLAMA_BUILD/qwen3-reference"
python3 tests/models/qwen3_checkpoint_parity.py build "$OFFICIAL_GGUF" "$LLAMA_BUILD/qwen3-reference" report.json
```

The reference helper calls llama.cpp tokenization with both add-special and
parse-special enabled, then runs its own last-pooled embeddings. It explicitly
uses F32 K/V cache and standard CPU buffers (no CPU_REPACK), matching native
full-sequence attention and mmap-backed operands. Optional `--f32-operands`
uses the upstream scheduler callback to decode only the current block-matmul
operand to F32 and releases it immediately afterward. This test-only arithmetic
adapter matches the public C API's `CLAP_ACTIVATION_F32`; it neither converts a
model file nor implements Qwen quantization, and the upstream model graph and
backend remain independent of clap. It never calls
clap. Its process exits before native inference starts, avoiding simultaneous
model residency. The harness checks exact token IDs and native C API text/token
embeddings, then CLI embeddings, across English, Chinese, combining accents,
emoji, contractions, whitespace and special-token inputs; it records model
size/SHA-256 and numerical errors, including raw native/reference vectors,
cosine and relative L2. The optional harness accepts `--dimension 2560` for 4B
or `--dimension 4096` for 8B, `--context 40960`, an expected `--sha256`, source
`--revision`, and an optional `--texts probes.json` corpus for bounded runs.
Use `--activation-precision model-default` when testing packed quantized C API
arithmetic. For matched-F32 C API validation, pass `--reference-f32-operands
--api-only`; the report explicitly marks the default CLI as unchecked, since
its arithmetic policy is different. Raw C API vectors are saved even on a strict
numerical failure. The defaults retain the original 0.6B nine-input corpus.
GPU and maximum-context execution are not covered by this CPU harness.

Previous 0.6B verification, before the explicit F32-KV/no-repack oracle settings,
on the pinned official F16 file above: static bundled-ICU CTest 36/36
and shared system-ICU CTest 35/35 passed. Both builds matched llama.cpp token
IDs for all nine published-model corpus entries. CLI embeddings had worst
absolute coordinate error 0.000879910 and minimum cosine similarity
0.999998666; C API F32-activation text/token embeddings had worst absolute
coordinate error 0.000097471. These compare the same GGUF and the same token
IDs, not differently normalized HF-tokenizer inputs.

4B/8B published-checkpoint validation
-----------------------------------

Both immutable official Q4_K_M files listed above were downloaded, checked
against their published SHA-256 digests and executed on CPU. The existing
parameterized runtime needed **no production-code changes**. Native CLI
inference returned finite, unit-normalized vectors of 2560 and 4096 elements.
After the final source freeze, reconfigured Release builds passed all registered
CTest checks: static/bundled ICU 36/36 and shared/system ICU 35/35, including the
expanded Qwen3 scalar/metadata suite. Tracked C/C++ formatting was checked with
clang-format 18.1.8; README, production sources, converters and quantizers are
unchanged.

Matched-F32 arithmetic is checked separately from default packed arithmetic:

| Checkpoint | Inputs | C API text/token max absolute error | Minimum cosine | Maximum relative L2 |
| --- | --- | ---: | ---: | ---: |
| 4B Q4_K_M | `hello`, `中文🙂` | 1.081e-7 | 0.9999999999989 | 1.452e-6 |
| 8B Q4_K_M | `hello` | 1.286e-7 | 0.9999999999995 | 9.237e-7 |

These passes compare the same published quantized weights with both sides
performing F32 block-matmul arithmetic, exact tokenizer IDs and full output
widths. The reference uses the optional per-operand adapter described above,
not a converted F32 checkpoint. Its storage-rounded F16 tiny-model result was
also checked against the independent scalar oracle. Reproduce this C API gate
with the optional harness, for example:

```sh
python3 tests/models/qwen3_checkpoint_parity.py build "$OFFICIAL_4B_GGUF" \
  "$LLAMA_BUILD/qwen3-reference" report-4b.json \
  --dimension 2560 --context 40960 --texts probes.json \
  --sha256 2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b \
  --revision f4602530db1d980e16da9d7d3a70294cf5c190be \
  --reference-f32-operands --api-only
```

**The default packed-arithmetic strict parity gates did not pass.** On `hello`,
native default CLI versus the F32-KV/no-repack llama reference retained these
measured differences: 4B max absolute error 0.00548702, cosine 0.9995348796,
relative L2 0.03049986; 8B max absolute error 0.00253895, cosine 0.9994889457,
relative L2 0.03197044. Both fail the unchanged cosine >0.9999/relative-L2 <0.02
checks; the 4B coordinate check also fails. The original repacked/F16-KV oracle
failure is retained separately, not relabeled as passing.

Layer-by-layer tracing on 4B localized the first material jump to block 16's
attention-output activation packing. Before packing, inputs differ by at most
8.047e-7. A shared value 0.09256741404533386 is packed to -12 versus -11 when
the Q8_K scale differs by one F32 ULP. The two independently linked GGML
packers produce bit-identical bytes on identical captured inputs; nearby inputs
produce different packed bytes, then later blocks amplify the discrepancy.
Disabling CPU_REPACK alone does not eliminate it. Removing activation packing
on both sides restores the matched-F32 results above, without changing the
weights, loader, attention implementation or tolerances. This distinguishes
activation-quantization threshold amplification from a missing size/layout
implementation. It does not establish retrieval-quality equivalence, GPU
parity, maximum-context behavior, or strict default-CLI numerical equivalence.
