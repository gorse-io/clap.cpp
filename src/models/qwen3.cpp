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
    config = nlohmann::json::parse(loader.metadata("qwen3.config"));
    const auto positive = [&](const char *key) {
        const auto &v = config.at(key);
        if (!v.is_number_integer() || v.get<int64_t>() <= 0 || v.get<int64_t>() > (1 << 20))
            throw std::runtime_error(std::string("CLAP: invalid Qwen3 ") + key);
        return v.get<int>();
    };
    h.width = positive("hidden_size");
    h.ffn = positive("intermediate_size");
    h.depth = positive("num_hidden_layers");
    h.heads = positive("num_attention_heads");
    h.kv_heads = positive("num_key_value_heads");
    h.head_dim = positive("head_dim");
    h.vocab = positive("vocab_size");
    h.context = positive("max_position_embeddings");
    if (config.at("model_type") != "qwen3" || config.at("hidden_act") != "silu" ||
        config.value("attention_bias", false) || config.value("use_sliding_window", false) ||
        !config.value("rope_scaling", nlohmann::json()).is_null())
        throw std::runtime_error("CLAP: unsupported Qwen3 activation/bias/sliding window/RoPE scaling");
    if (h.heads % h.kv_heads || h.head_dim % 2 || int64_t(h.heads) * h.head_dim > (1 << 20))
        throw std::runtime_error("CLAP: invalid Qwen3 GQA/head dimension");
    h.epsilon = config.at("rms_norm_eps").get<float>();
    h.theta = config.at("rope_theta").get<float>();
    if (!std::isfinite(h.epsilon) || h.epsilon <= 0 || !std::isfinite(h.theta) || h.theta <= 0)
        throw std::runtime_error("CLAP: invalid Qwen3 RMS epsilon/RoPE base");
    const auto token_id = [&](const char *key, int fallback) {
        if (!config.contains(key) || config.at(key).is_null())
            return fallback;
        const auto &v = config.at(key);
        if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() >= h.vocab)
            throw std::runtime_error(std::string("CLAP: invalid Qwen3 ") + key);
        return v.get<int>();
    };
    h.eos = token_id("eos_token_id", -1);
    if (h.eos < 0)
        throw std::runtime_error("CLAP: Qwen3 requires an EOS token");
    token_id("bos_token_id", -1);
    h.pad = token_id("pad_token_id", -1);
    // The published checkpoint uses EOS for external batch padding. With no
    // attention-mask API, those inputs must be unpadded to retain the real EOS.
    if (h.pad == h.eos)
        h.pad = -1;
}

void Qwen3Model::load_arch_tensors() const {
    loader.shape("model.embed_tokens.weight", {h.width, h.vocab});
    loader.shape("model.norm.weight", {h.width});
    for (int i = 0; i < h.depth; ++i) {
        const auto p = "model.layers." + std::to_string(i) + ".";
        for (const char *name : {"input_layernorm", "post_attention_layernorm"})
            loader.shape(p + name + ".weight", {h.width});
        for (const char *name : {"q_norm", "k_norm"})
            loader.shape(p + "self_attn." + name + ".weight", {h.head_dim});
        loader.linear_shape(p + "self_attn.q_proj", h.width, h.heads * h.head_dim, false);
        loader.linear_shape(p + "self_attn.k_proj", h.width, h.kv_heads * h.head_dim, false);
        loader.linear_shape(p + "self_attn.v_proj", h.width, h.kv_heads * h.head_dim, false);
        loader.linear_shape(p + "self_attn.o_proj", h.heads * h.head_dim, h.width, false);
        loader.linear_shape(p + "mlp.gate_proj", h.width, h.ffn, false);
        loader.linear_shape(p + "mlp.up_proj", h.width, h.ffn, false);
        loader.linear_shape(p + "mlp.down_proj", h.ffn, h.width, false);
    }
}
void Qwen3Model::bind_tensors() {
    const auto bind = [&](const std::string &name) { return LayerWeights::bind(loader, name, false); };
    embeddings = loader.weight("model.embed_tokens.weight");
    final_norm = bind("model.norm");
    for (int i = 0; i < h.depth; ++i) {
        const auto p = "model.layers." + std::to_string(i) + ".";
        layers.push_back({bind(p + "input_layernorm"), bind(p + "post_attention_layernorm"),
                          bind(p + "self_attn.q_norm"), bind(p + "self_attn.k_norm"),
                          bind(p + "self_attn.q_proj"), bind(p + "self_attn.k_proj"),
                          bind(p + "self_attn.v_proj"), bind(p + "self_attn.o_proj"),
                          bind(p + "mlp.gate_proj"), bind(p + "mlp.up_proj"), bind(p + "mlp.down_proj")});
    }
}
ggml_tensor *Qwen3Model::rms(Graph &g, ggml_tensor *x, const LayerWeights &norm) const {
    return ggml_mul(g.ctx(), ggml_rms_norm(g.ctx(), x, h.epsilon), g.weight(norm.weight));
}
ggml_tensor *Qwen3Model::expand_kv(Graph &g, ggml_tensor *x, int tokens) const {
    // Repeat each KV head contiguously (0,0,1,1), not modulo the head count.
    auto *split = ggml_reshape_4d(g.ctx(), x, h.head_dim, 1, h.kv_heads, tokens);
    auto *repeated = ggml_repeat_4d(g.ctx(), split, h.head_dim, h.heads / h.kv_heads, h.kv_heads, tokens);
    return ggml_reshape_2d(g.ctx(), repeated, h.heads * h.head_dim, tokens);
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
    return runtime.run_cached("qwen3-text", {GraphInput(tokens)}, [&](Graph &g) {
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
            auto *attn = g.attention(q, expand_kv(g, k, n), expand_kv(g, v, n), h.heads, causal);
            x = ggml_add(ctx, x, g.linear(attn, layer.output));
            z = rms(g, x, layer.post_norm);
            auto *gated = ggml_mul(ctx, ggml_silu(ctx, g.linear(z, layer.gate)), g.linear(z, layer.up));
            x = ggml_add(ctx, x, g.linear(gated, layer.down));
        }
        x = rms(g, x, final_norm);
        auto *last = ggml_cont(ctx, ggml_view_1d(ctx, x, h.width, size_t(n - 1) * x->nb[1]));
        return std::vector<ggml_tensor *>{g.l2_normalize(last)};
    }).front().values;
}
AudioFrames Qwen3Model::audio_frames(const std::vector<float> &) const {
    throw std::runtime_error("CLAP: Qwen3 has no audio encoder");
}
} // namespace clap::detail
