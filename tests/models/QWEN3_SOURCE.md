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

Metadata and tensors
--------------------

The model uses typed `qwen3.*` hyperparameters and `tokenizer.ggml.*` arrays.
Integer dimensions/IDs are UINT32, RMS epsilon/RoPE base are FLOAT32, tokenizer
flags are BOOL, tokens/merges are STRING arrays and token types are INT32 arrays.
Tensor names are `token_embd.weight`, `output_norm.weight`, and `blk.N.*`
(`attn_norm`, `attn_q/k/v`, `attn_output`, `attn_q/k_norm`, `ffn_norm`,
`ffn_gate/up/down`). An optional correctly shaped `output.weight` is accepted
but unused for embeddings. Unknown, missing and incorrectly shaped tensors fail.

The official model has hidden width 1024, FFN width 3072, 28 blocks, 16 query
heads, 8 KV heads, head dimension 128, context 32768 and vocabulary 151669.
Query width 2048 is independent of hidden width. The runtime retains causal
contiguous GQA, weighted per-head Q/K RMSNorm, half-split RoPE with base 1e6,
pre-RMSNorm residual blocks, SiLU gated FFN, final RMSNorm and last-token/L2
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
blocks, asymmetric head/hidden dimensions, GQA, causal masking, Q/K RMSNorm,
RoPE and pooling. It also covers Unicode without NFC, literal special tokens,
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
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" -lllama -o "$LLAMA_BUILD/qwen3-reference"
python3 tests/models/qwen3_checkpoint_parity.py build "$OFFICIAL_GGUF" "$LLAMA_BUILD/qwen3-reference" report.json
```

The reference helper calls llama.cpp tokenization with both add-special and
parse-special enabled, then runs its own last-pooled embeddings. It never calls
clap. Its process exits before native inference starts, avoiding simultaneous
model residency. The harness checks exact token IDs and native C API text/token
embeddings, then CLI embeddings, across English, Chinese, combining accents,
emoji, contractions, whitespace and special-token inputs; it records model
size/SHA-256 and numerical errors. GPU and maximum-context execution are not
covered by this CPU harness.

Verified on the pinned official F16 file above: static bundled-ICU CTest 36/36
and shared system-ICU CTest 35/35 passed. Both builds matched llama.cpp token
IDs for all nine published-model corpus entries. CLI embeddings had worst
absolute coordinate error 0.000879910 and minimum cosine similarity
0.999998666; C API F32-activation text/token embeddings had worst absolute
coordinate error 0.000097471. These compare the same GGUF and the same token
IDs, not differently normalized HF-tokenizer inputs.
