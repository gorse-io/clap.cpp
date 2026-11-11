#pragma once
#include <ggml.h>
namespace clap::detail {
// Q: F32 [dim,tokens,heads], K/V: F16 [dim,tokens,kv_heads].
// Output: F32 [dim,heads,tokens]. CPU-only, causal, no KV cache.
ggml_tensor *qwen3_attention_cpu(ggml_context *ctx, ggml_tensor *q, ggml_tensor *k, ggml_tensor *v, ggml_tensor *mask);
} // namespace clap::detail
