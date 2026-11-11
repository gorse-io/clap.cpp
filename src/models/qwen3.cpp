#include "models.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace clap::detail {
Qwen3Model::Qwen3Model(const ModelLoader &source, int threads) : loader(source), runtime(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
    bind_tensors();
}

void Qwen3Model::load_arch_hparams() {
    if (loader.has_metadata("qwen3.config") || loader.has_metadata("qwen3.tokenizer"))
        throw std::runtime_error("CLAP: Qwen3 requires standard llama.cpp GGUF, not the legacy native format");
    const auto positive = [&](const char *key) {
        const auto value = loader.metadata_u32(key);
        if (value == 0 || value > (1u << 20))
            throw std::runtime_error(std::string("CLAP: invalid Qwen3 ") + key);
        return int(value);
    };
    h.width = positive("qwen3.embedding_length");
    h.ffn = positive("qwen3.feed_forward_length");
    h.depth = positive("qwen3.block_count");
    h.heads = positive("qwen3.attention.head_count");
    h.kv_heads = positive("qwen3.attention.head_count_kv");
    h.head_dim =
        loader.has_metadata("qwen3.attention.key_length") ? positive("qwen3.attention.key_length") : h.width / h.heads;
    h.context = positive("qwen3.context_length");
    h.vocab = int(loader.weight("token_embd.weight")->ne[1]);
    if (h.vocab <= 0 || h.vocab > (1 << 24) || h.heads % h.kv_heads || h.head_dim % 2 ||
        int64_t(h.heads) * h.head_dim > (1 << 20) ||
        (loader.has_metadata("qwen3.attention.value_length") &&
         positive("qwen3.attention.value_length") != h.head_dim) ||
        (loader.has_metadata("qwen3.rope.dimension_count") && positive("qwen3.rope.dimension_count") != h.head_dim))
        throw std::runtime_error("CLAP: invalid Qwen3 GQA/head dimension");
    h.epsilon = loader.metadata_f32("qwen3.attention.layer_norm_rms_epsilon");
    h.theta = loader.has_metadata("qwen3.rope.freq_base") ? loader.metadata_f32("qwen3.rope.freq_base") : 10000.0f;
    if (!std::isfinite(h.epsilon) || h.epsilon <= 0 || !std::isfinite(h.theta) || h.theta <= 0)
        throw std::runtime_error("CLAP: invalid Qwen3 RMS epsilon/RoPE base");
    if ((loader.has_metadata("qwen3.pooling_type") && loader.metadata_u32("qwen3.pooling_type") != 3) ||
        (loader.has_metadata("qwen3.attention.causal") && !loader.metadata_bool("qwen3.attention.causal")) ||
        (loader.has_metadata("qwen3.rope.scaling.type") && loader.metadata("qwen3.rope.scaling.type") != "none") ||
        loader.has_metadata("qwen3.attention.sliding_window"))
        throw std::runtime_error("CLAP: unsupported Qwen3 pooling/attention/RoPE scaling");
    const auto token_id = [&](const char *key) {
        if (!loader.has_metadata(key))
            return -1;
        const auto value = loader.metadata_u32(key);
        if (value >= uint32_t(h.vocab))
            throw std::runtime_error(std::string("CLAP: invalid Qwen3 ") + key);
        return int(value);
    };
    h.eos = token_id("tokenizer.ggml.eos_token_id");
    token_id("tokenizer.ggml.bos_token_id");
    h.pad = token_id("tokenizer.ggml.padding_token_id");
    if (h.pad == h.eos)
        h.pad = -1;
}

void Qwen3Model::load_arch_tensors() const {
    loader.shape("token_embd.weight", {h.width, h.vocab});
    loader.shape("output_norm.weight", {h.width});
    if (loader.has_tensor_prefix("output.weight"))
        loader.shape("output.weight", {h.width, h.vocab});
    for (int i = 0; i < h.depth; ++i) {
        const auto p = "blk." + std::to_string(i) + ".";
        for (const char *name : {"attn_norm", "ffn_norm"})
            loader.shape(p + name + ".weight", {h.width});
        for (const char *name : {"attn_q_norm", "attn_k_norm"})
            loader.shape(p + name + ".weight", {h.head_dim});
        loader.linear_shape(p + "attn_q", h.width, h.heads * h.head_dim, false);
        loader.linear_shape(p + "attn_k", h.width, h.kv_heads * h.head_dim, false);
        loader.linear_shape(p + "attn_v", h.width, h.kv_heads * h.head_dim, false);
        loader.linear_shape(p + "attn_output", h.heads * h.head_dim, h.width, false);
        loader.linear_shape(p + "ffn_gate", h.width, h.ffn, false);
        loader.linear_shape(p + "ffn_up", h.width, h.ffn, false);
        loader.linear_shape(p + "ffn_down", h.ffn, h.width, false);
    }
}
void Qwen3Model::bind_tensors() {
    const auto bind = [&](const std::string &name) { return LayerWeights::bind(loader, name, false); };
    embeddings = loader.weight("token_embd.weight");
    final_norm = bind("output_norm");
    for (int i = 0; i < h.depth; ++i) {
        const auto p = "blk." + std::to_string(i) + ".";
        layers.push_back({bind(p + "attn_norm"), bind(p + "ffn_norm"), bind(p + "attn_q_norm"), bind(p + "attn_k_norm"),
                          bind(p + "attn_q"), bind(p + "attn_k"), bind(p + "attn_v"), bind(p + "attn_output"),
                          bind(p + "ffn_gate"), bind(p + "ffn_up"), bind(p + "ffn_down")});
    }
}
ggml_tensor *Qwen3Model::rms(Graph &g, ggml_tensor *x, const LayerWeights &norm) const {
    return ggml_mul(g.ctx(), ggml_rms_norm(g.ctx(), x, h.epsilon), g.weight(norm.weight));
}

std::vector<float> Qwen3Model::text(const std::vector<int> &ids) const {
    if (ids.empty() || ids.size() > size_t(h.context))
        throw std::runtime_error("CLAP: Qwen3 token count is outside context length");
    std::vector<int> tokens;
    tokens.reserve(ids.size());
    for (const int id : ids) {
        if (id < 0 || id >= h.vocab)
            throw std::runtime_error("CLAP: Qwen3 embedding index out of range");
        if (id != h.pad)
            tokens.push_back(id);
    }
    if (tokens.empty())
        throw std::runtime_error("CLAP: Qwen3 requires a nonpadding token");
    // Compact masked pads before execution: valid positions stay contiguous,
    // and the final column is exactly the last nonpadding query.
    return runtime
        .run_cached("qwen3-text", {GraphInput(tokens)},
                    [&](Graph &g) {
                        const int n = int(tokens.size());
                        auto *ctx = g.ctx();
                        std::vector<float> mask(size_t(n) * n, 0.0f);
                        for (int row = 0; row < n; ++row)
                            for (int col = row + 1; col < n; ++col)
                                mask[size_t(row) * n + col] = -std::numeric_limits<float>::infinity();
                        auto *causal = g.input(mask, n, n);
                        auto *x = g.embedding(embeddings, g.dynamic_input(0));
                        for (const auto &layer : layers) {
                            auto *z = rms(g, x, layer.input_norm);
                            auto *q = ggml_reshape_3d(ctx, g.linear(z, layer.query), h.head_dim, h.heads, n);
                            auto *k = ggml_reshape_3d(ctx, g.linear(z, layer.key), h.head_dim, h.kv_heads, n);
                            q = ggml_reshape_2d(ctx, rms(g, q, layer.q_norm), h.head_dim * h.heads, n);
                            k = ggml_reshape_2d(ctx, rms(g, k, layer.k_norm), h.head_dim * h.kv_heads, n);
                            q = g.rope(q, h.heads, h.theta);
                            k = g.rope(k, h.kv_heads, h.theta);
                            auto *v = g.linear(z, layer.value);
                            auto *attn = g.qwen3_attention(q, k, v, h.heads, h.kv_heads, causal);
                            x = ggml_add(ctx, x, g.linear(attn, layer.output));
                            z = rms(g, x, layer.post_norm);
                            auto *gated = ggml_mul(ctx, ggml_silu(ctx, g.linear(z, layer.gate)), g.linear(z, layer.up));
                            x = ggml_add(ctx, x, g.linear(gated, layer.down));
                        }
                        x = rms(g, x, final_norm);
                        auto *last = ggml_cont(ctx, ggml_view_1d(ctx, x, h.width, size_t(n - 1) * x->nb[1]));
                        return std::vector<ggml_tensor *>{g.l2_normalize(last)};
                    })
        .front()
        .values;
}
AudioFrames Qwen3Model::audio_frames(const std::vector<float> &) const {
    throw std::runtime_error("CLAP: Qwen3 has no audio encoder");
}
} // namespace clap::detail
