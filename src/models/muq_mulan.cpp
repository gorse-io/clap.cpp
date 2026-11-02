#include "models.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace clap::detail {
namespace {
void positive(const nlohmann::json &obj, const char *key) {
    if (!obj.at(key).is_number_integer() || obj.at(key).get<int64_t>() <= 0 || obj.at(key).get<int64_t>() > 10000000)
        throw std::runtime_error(std::string("CLAP: invalid MuLan dimension: ") + key);
}
} // namespace
MuqMulanModel::MuqMulanModel(const ModelLoader &source, int threads)
    : loader(source), runtime(source, threads), audio_backbone(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
    bind_tensors();
}
void MuqMulanModel::load_arch_hparams() {
    config = nlohmann::json::parse(loader.metadata("muq_mulan.config"));
    text_config = nlohmann::json::parse(loader.metadata("muq_mulan.text_config"));
    const auto &m = config.at("mulan"), &a = config.at("audio_transformer"), &t = config.at("text_transformer");
    positive(m, "clip_secs");
    positive(m, "dim_latent");
    if (m.at("clip_secs").get<int64_t>() > 3600 || !m.at("sr").is_number_integer() || m.at("sr") != 24000 ||
        m.value("hierarchical_contrastive_loss", false) ||
        config.at("audio_model").at("name") != "OpenMuQ/MuQ-large-msd-iter" ||
        !config.at("audio_model").at("use_layer_idx").is_number_integer() ||
        config.at("audio_model").at("use_layer_idx") != -1 ||
        !config.at("audio_model").at("model_dim").is_number_integer() ||
        config.at("audio_model").at("model_dim") != audio_backbone.projection_dimension() || a.at("tf_depth") != 0 ||
        config.at("text_model").at("name") != "xlm-roberta-base")
        throw std::runtime_error("CLAP: unsupported MuLan backbone/layer/clip variant");
    for (const auto *branch : {&a, &t}) {
        for (const auto *key : {"dim", "heads", "dim_head"})
            positive(*branch, key);
        const double mult = branch->at("ff_mult").get<double>();
        const int64_t inner = branch->at("heads").get<int64_t>() * branch->at("dim_head").get<int64_t>();
        const double ff = branch->at("dim").get<double>() * mult * 2 / 3;
        if (!(inner <= 1048576 && std::isfinite(ff) && ff >= 1 && ff <= 1048576 && std::isfinite(mult) && mult > 0 &&
              mult <= 100 && branch->at("tf_depth").is_number_integer() && branch->at("tf_depth").get<int64_t>() >= 0 &&
              branch->at("tf_depth").get<int64_t>() <= 1000))
            throw std::runtime_error("CLAP: invalid MuLan adapter dimensions");
    }
    for (const auto *key : {"hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                            "vocab_size", "max_position_embeddings", "type_vocab_size"})
        positive(text_config, key);
    const float eps = text_config.at("layer_norm_eps");
    if (t.at("dim") != text_config.at("hidden_size") ||
        !(config.at("text_model").at("model_dim").is_null() ||
          (config.at("text_model").at("model_dim").is_number_integer() &&
           config.at("text_model").at("model_dim") == text_config.at("hidden_size"))) ||
        text_config.at("model_type") != "xlm-roberta" || text_config.at("hidden_act") != "gelu" ||
        text_config.value("position_embedding_type", std::string("absolute")) != "absolute" ||
        text_config.value("is_decoder", false) || text_config.value("add_cross_attention", false) ||
        text_config.at("hidden_size").get<int>() % text_config.at("num_attention_heads").get<int>() != 0 ||
        !text_config.at("pad_token_id").is_number_integer() || !text_config.at("bos_token_id").is_number_integer() ||
        !text_config.at("eos_token_id").is_number_integer() || text_config.at("pad_token_id") != 1 ||
        text_config.at("bos_token_id") != 0 || text_config.at("eos_token_id") != 2 ||
        text_config.at("max_position_embeddings").get<int>() < 4 || text_config.at("vocab_size").get<int>() < 4 ||
        !(std::isfinite(eps) && eps > 0))
        throw std::runtime_error("CLAP: unsupported MuLan XLM-R configuration");
    const auto tokenizer = nlohmann::json::parse(loader.metadata("muq_mulan.tokenizer"));
    const auto &vocab = tokenizer.at("model").at("vocab");
    if (tokenizer.at("model").value("type", std::string("Unigram")) != "Unigram" ||
        vocab.size() != text_config.at("vocab_size").get<size_t>() || vocab.at(0).at(0) != "<s>" ||
        vocab.at(1).at(0) != "<pad>" || vocab.at(2).at(0) != "</s>" || vocab.at(3).at(0) != "<unk>")
        throw std::runtime_error("CLAP: MuLan tokenizer/config mismatch");
}
void MuqMulanModel::load_arch_tensors() const {
    const int d = text_config.at("hidden_size"), dim = config.at("text_transformer").at("dim");
    loader.shape("text.embeddings.word_embeddings.weight", {d, text_config.at("vocab_size").get<int>()});
    loader.shape("text.embeddings.position_embeddings.weight",
                 {d, text_config.at("max_position_embeddings").get<int>()});
    loader.shape("text.embeddings.token_type_embeddings.weight", {d, text_config.at("type_vocab_size").get<int>()});
    loader.norm_shape("text.embeddings.LayerNorm", d);
    for (int i = 0; i < text_config.at("num_hidden_layers").get<int>(); ++i) {
        const auto p = "text.encoder.layer." + std::to_string(i);
        for (const auto *name : {"query", "key", "value"})
            loader.linear_shape(p + ".attention.self." + name, d, d);
        loader.linear_shape(p + ".attention.output.dense", d, d);
        loader.norm_shape(p + ".attention.output.LayerNorm", d);
        loader.linear_shape(p + ".intermediate.dense", d, text_config.at("intermediate_size"));
        loader.linear_shape(p + ".output.dense", text_config.at("intermediate_size"), d);
        loader.norm_shape(p + ".output.LayerNorm", d);
    }
    const auto &t = config.at("text_transformer");
    const int inner = t.at("heads").get<int>() * t.at("dim_head").get<int>();
    const int ff = static_cast<int>(dim * t.at("ff_mult").get<double>() * 2 / 3);
    if (ff <= 0)
        throw std::runtime_error("CLAP: invalid MuLan GEGLU width");
    for (int i = 0; i < t.at("tf_depth").get<int>(); ++i) {
        const auto p = "adapter.layers." + std::to_string(i);
        loader.shape(p + ".0.norm.learned_gamma", {dim});
        loader.shape(p + ".0.q_scale", {t.at("dim_head").get<int>()});
        loader.shape(p + ".0.k_scale", {t.at("dim_head").get<int>()});
        loader.linear_shape(p + ".0.to_q", dim, inner, false);
        loader.linear_shape(p + ".0.to_kv", dim, 2 * inner, false);
        loader.linear_shape(p + ".0.to_out.0", inner, dim, false);
        loader.shape(p + ".1.0.learned_gamma", {dim});
        loader.linear_shape(p + ".1.1", dim, 2 * ff, false);
        loader.linear_shape(p + ".1.4", ff, dim, false);
    }
    loader.linear_shape("audio_proj", audio_backbone.projection_dimension(), config.at("audio_transformer").at("dim"));
    loader.linear_shape("audio_latent", config.at("audio_transformer").at("dim"),
                        config.at("mulan").at("dim_latent").get<int>());
    loader.linear_shape("text_proj", dim, config.at("mulan").at("dim_latent").get<int>());
}

void MuqMulanModel::bind_tensors() {
    h = {config.at("mulan").at("dim_latent"),
         text_config.at("max_position_embeddings"),
         text_config.at("vocab_size"),
         text_config.at("num_attention_heads"),
         config.at("text_transformer").at("heads"),
         config.at("text_transformer").at("dim_head"),
         config.at("mulan").at("clip_secs"),
         text_config.at("layer_norm_eps")};
    embeddings = bind_embeddings(loader, "text.embeddings");
    for (int i = 0; i < text_config.at("num_hidden_layers").get<int>(); ++i)
        text_layers.push_back(bind_bert_layer(loader, "text.encoder.layer." + std::to_string(i)));
    for (int i = 0; i < config.at("text_transformer").at("tf_depth").get<int>(); ++i) {
        const auto p = "adapter.layers." + std::to_string(i);
        adapter_layers.push_back(
            {loader.weight(p + ".0.norm.learned_gamma"), loader.weight(p + ".1.0.learned_gamma"),
             loader.weight(p + ".0.q_scale"), loader.weight(p + ".0.k_scale"),
             LayerWeights::bind(loader, p + ".0.to_q", false), LayerWeights::bind(loader, p + ".0.to_kv", false),
             LayerWeights::bind(loader, p + ".0.to_out.0", false), LayerWeights::bind(loader, p + ".1.1", false),
             LayerWeights::bind(loader, p + ".1.4", false)});
    }
    text_projection = LayerWeights::bind(loader, "text_proj");
    audio_projection = LayerWeights::bind(loader, "audio_proj");
    audio_latent = LayerWeights::bind(loader, "audio_latent");
}

TextParameters MuqMulanModel::text_parameters() const {
    // Tokenize without truncation; reject over-context input in text(), since
    // the official wrapper also does not truncate or chunk a single item.
    return {0, 2, std::numeric_limits<size_t>::max()};
}
int MuqMulanModel::projection_dimension() const { return h.projection; }
AudioFrames MuqMulanModel::audio_frames(const std::vector<float> &) const {
    throw std::runtime_error("CLAP: MuLan exposes embeddings, not frames");
}
ggml_tensor *MuqMulanModel::encode_text(Graph &g, ggml_tensor *embeddings) const {
    auto *ctx = g.ctx();
    const int heads = h.heads;
    const float eps = h.epsilon;
    auto *x = g.norm(embeddings, this->embeddings.norm, eps);
    for (const auto &layer : text_layers) {
        auto *q = g.linear(x, layer.attention.query), *k = g.linear(x, layer.attention.key),
             *v = g.linear(x, layer.attention.value);
        auto *y = g.linear(g.attention(q, k, v, heads), layer.attention.output);
        x = g.norm(ggml_add(ctx, y, x), layer.attention_norm, eps);
        y = ggml_gelu_erf(ctx, g.linear(x, layer.intermediate));
        y = g.linear(y, layer.output);
        x = g.norm(ggml_add(ctx, y, x), layer.output_norm, eps);
    }
    return x;
}
ggml_tensor *MuqMulanModel::biasless_norm(Graph &g, ggml_tensor *x, ggml_tensor *weight) const {
    return ggml_mul(g.ctx(), ggml_norm(g.ctx(), x, 1e-5f), g.weight(weight));
}
ggml_tensor *MuqMulanModel::adapter(Graph &g, ggml_tensor *x) const {
    auto *ctx = g.ctx();
    const int heads = h.adapter_heads, hd = h.head_dim, inner = heads * hd;
    for (const auto &layer : adapter_layers) {
        auto *normalized = biasless_norm(g, x, layer.norm);
        auto *q = g.linear(normalized, layer.query), *kv = g.linear(normalized, layer.key_value);
        auto *k = g.slice(kv, 0, inner), *v = g.slice(kv, inner, inner);
        auto normalize_heads = [&](ggml_tensor *value, ggml_tensor *weight) {
            value = ggml_reshape_3d(ctx, value, hd, heads, x->ne[1]);
            value = ggml_mul(ctx, g.l2_normalize(value), g.weight(weight));
            return ggml_reshape_2d(ctx, value, inner, x->ne[1]);
        };
        q = normalize_heads(q, layer.q_scale);
        k = normalize_heads(k, layer.k_scale);
        auto *attended = g.attention(q, k, v, heads, nullptr, 8.0f);
        x = ggml_add(ctx, x, g.linear(attended, layer.output));
        auto *y = g.linear(biasless_norm(g, x, layer.ffn_norm), layer.ffn_input);
        const int64_t ff = y->ne[0] / 2;
        y = ggml_mul(ctx, g.slice(y, 0, ff), ggml_gelu_erf(ctx, g.slice(y, ff, ff)));
        x = ggml_add(ctx, x, g.linear(y, layer.ffn_output));
    }
    return x;
}
ggml_tensor *MuqMulanModel::project(Graph &g, ggml_tensor *x, const LayerWeights &weights) const {
    return g.l2_normalize(g.linear(g.mean_pool(x), weights));
}
std::vector<float> MuqMulanModel::text(const std::vector<int> &ids) const {
    const int positions = h.positions, vocab = h.vocab;
    if (ids.empty() || ids.size() > static_cast<size_t>(positions - 2))
        throw std::runtime_error("CLAP: MuLan text is empty or exceeds XLM-R context; truncation/chunking unsupported");
    std::vector<int> position_ids(ids.size());
    int position = 1;
    for (size_t r = 0; r < ids.size(); ++r) {
        const int id = ids[r];
        if (id < 0 || id >= vocab)
            throw std::runtime_error("CLAP: MuLan token ID out of vocabulary");
        const int p = id == 1 ? 1 : ++position;
        if (p >= positions)
            throw std::runtime_error("CLAP: MuLan text positions exceed context");
        position_ids[r] = p;
    }
    auto outputs = runtime.run_cached("text", {GraphInput(ids), GraphInput(position_ids)}, [&](Graph &g) {
        auto *x = ggml_add(g.ctx(), g.embedding(embeddings.word, g.dynamic_input(0)),
                           ggml_add(g.ctx(), g.embedding(embeddings.position, g.dynamic_input(1)),
                                    g.embedding(embeddings.type, std::vector<int>{0})));
        return std::vector<ggml_tensor *>{project(g, adapter(g, encode_text(g, x)), text_projection)};
    });
    return std::move(outputs.front().values);
}
std::vector<float> MuqMulanModel::waveform_embedding(const std::vector<float> &pcm, size_t crop_offset) const {
    if (crop_offset != 0)
        throw std::runtime_error("CLAP: crop offsets are unsupported for this model");
    if (pcm.empty() || pcm.size() > (size_t{1} << 28) ||
        !std::all_of(pcm.begin(), pcm.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("CLAP: MuLan requires finite nonempty waveform of at most 1 GiB");
    const size_t delta = 24000 * h.clip_seconds;
    std::vector<float> result(projection_dimension());
    size_t count = 0;
    for (size_t start = 0; start < pcm.size(); start += delta) {
        const size_t remaining = std::min(delta, pcm.size() - start);
        std::vector<float> clip(pcm.begin() + start, pcm.begin() + start + remaining);
        // Exactly one prefix slice, even when it cannot fill a ten-second clip.
        if (remaining < delta)
            clip.insert(clip.end(), pcm.begin(), pcm.begin() + std::min(delta - remaining, pcm.size()));
        auto frames = audio_backbone.audio_frames(clip);
        const auto outputs =
            runtime.run_cached("audio", {GraphInput(frames.values, frames.dimension, frames.frames)}, [&](Graph &g) {
                return std::vector<ggml_tensor *>{
                    project(g, g.linear(g.dynamic_input(0), audio_projection), audio_latent)};
            });
        const auto &vector = outputs.front().values;
        for (size_t d = 0; d < result.size(); ++d)
            result[d] += vector[d];
        ++count;
    }
    for (float &v : result)
        v /= count;
    return result;
}
} // namespace clap::detail
