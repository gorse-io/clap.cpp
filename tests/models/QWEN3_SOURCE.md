Qwen3 Embedding source contract
==============================

Implementation references (read at immutable revisions):

- llama.cpp [`202c1a229f5a83d1c9d758ed0f2d43760fa4bab0`](https://github.com/ggml-org/llama.cpp/tree/202c1a229f5a83d1c9d758ed0f2d43760fa4bab0):
  `src/models/qwen3.cpp` (`load_arch_tensors`, `build_backbone`) and
  `conversion/qwen.py` (`Qwen3Model` / `Qwen2Model` converter),
  `conversion/base.py` (`_set_vocab_gpt2`, `_try_set_pooling_type`),
  `src/llama-vocab.cpp` (Qwen BPE pretokenization).
  The backbone uses causal grouped query attention, separate per-head Q/K
  RMSNorm, NeoX (half split) RoPE, pre-RMSNorm residual blocks, SiLU gated FFN,
  and final RMSNorm. The LM head is irrelevant for embeddings.
- [`unsloth/Qwen3-Embedding-0.6B`, revision
  `f2fddb42505bde9feaf19f0967b01dce52e764c6`](https://huggingface.co/unsloth/Qwen3-Embedding-0.6B/tree/f2fddb42505bde9feaf19f0967b01dce52e764c6):
  `config.json`, `tokenizer.json`, `tokenizer_config.json`, `README.md`,
  `modules.json`, `1_Pooling/config.json`. The published configuration is
  width 1024, FFN 3072, 28 blocks, 16 Q heads, 8 KV heads, head dimension 128,
  RMS epsilon 1e-6, RoPE base 1e6, context 32768, vocabulary 151669.
  In particular Q width is 2048, independent of hidden width 1024.
  Sentence Transformers modules specify last-token pooling followed by L2
  normalization; the README's Transformers example uses the same calculation.

The actual unsloth tokenizer.json uses NFC normalization, Qwen's Unicode regex
(including single digits), GPT2 byte BPE, recognition of added tokens, and a
postprocessor appending `<|endoftext|>` (151643). It does not prepend BOS.
This differs from assuming a generic generation tokenizer with no terminal token.
The native converter preserves that tokenizer, including its postprocessor;
chat templates and retrieval instructions are never inserted. Supply desired
instructions as literal input yourself.

Native GGUF uses HF tensor names and embedded JSON metadata, following this
repository's converters. llama.cpp-format GGUF with renamed tensors/typed
metadata is not the native converter format. There is no llama.cpp dependency.
F32/F16 conversion streams F32/F16/BF16 safetensors in bounded chunks. The
published single-file model.safetensors is supported. Arbitrary sharded, scaled
RoPE, sliding-window, biased or multimodal variants are rejected.

The C API accepts one token sequence, without an attention-mask argument. When
config.json defines a distinct pad_token_id, native encoding masks that ID and
pools the last nonpadding token. The target config has no pad_token_id, and its
external tokenizer uses endoftext both as EOS and batch padding. Supply unpadded
sequences for this checkpoint: an EOS token cannot be distinguished from external
padding by token ID alone. Native text tokenization never pads.

Default regression tests generate GGUF and safetensors in their build/temp
folders. Their independent Python scalar decoder uses two nonzero deterministic
blocks, asymmetric Q/hidden widths and GQA. Mutating causal attention, Q/K norm,
RoPE, KV-head assignment or pooling demonstrably changes the expected result.
Storage tests reuse the existing `test-q8` canonical GGML storage decoder, then
compare the independent scalar backbone against the decoded BF16/Q8/K weights.
No binary fixture, model download, torch or transformers is needed.

Reproduce default verification with generated fixtures:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j1
ctest --test-dir build --output-on-failure
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON
cmake --build build-shared -j1
ctest --test-dir build-shared --output-on-failure
```

`qwen3_checkpoint_parity.py BUILD LOCAL_HF_DIRECTORY REPORT.json` is an optional
real-checkpoint comparison using upstream Transformers Qwen3 modules. Put native
`model-f16.gguf` beside the local safetensors/config/tokenizer. The harness gathers
only the required embedding rows and loads one decoder layer at a time in F32.
It checks the actual exported tokenizer against Hugging Face AutoTokenizer and
compares all 1024 output coordinates. The reference worker processes all four
short texts for each loaded layer, then exits before native inference begins.

Published-checkpoint CPU comparison: native F16 GGUF against Transformers 4.53.2
Qwen3 modules with F32 arithmetic (torch 2.6.0+cpu), for English, Chinese,
combining-accent/emoji and literal-special-token texts, returned 1024-dimensional
unit vectors. Worst maximum absolute coordinate error was 0.0001178831; minimum
cosine similarity was 0.9999993997. GPU execution and maximum-context inference
were not exercised. F32/BF16/Q8_0/Q4_K_M storage arithmetic is covered separately
by the generated default fixtures.

The recorded default suites passed 38/38 static and 37/37 shared tests. Static
used pinned bundled ICU 78.3; the shared run added `-DCLAP_USE_SYSTEM_ICU=ON`
(system ICU 78.2) to the shared configure command above. Both were Release
builds compiled serially with `-j1`.
