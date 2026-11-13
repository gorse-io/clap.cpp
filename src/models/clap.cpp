#include "audio.h"
#include "model-policy.h"
#include "models.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace clap::detail {
TensorLoadPolicy ClapModel::loading_policy(const ModelLoader &) { return {}; }

using json = nlohmann::json;
namespace {
// PyTorch bicubic interpolation, A=-0.75, align_corners=True.
float cubic(float x) {
    x = std::abs(x);
    if (x <= 1)
        return ((1.25f * x - 2.25f) * x) * x + 1;
    if (x < 2)
        return ((-0.75f * x + 3.75f) * x - 6) * x + 3;
    return 0;
}
} // namespace

ClapModel::ClapModel(const ModelLoader &source, int threads) : loader(source), runtime(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
}

void ClapModel::load_arch_hparams() {
    config = json::parse(loader.metadata("clap.config"));
    a = config.at("audio_config");
    t = config.at("text_config");

    if (config.at("model_type") != "clap")
        throw std::runtime_error("CLAP: not a CLAP model");
    if (!(!a.at("enable_fusion").get<bool>()))
        throw std::runtime_error("CLAP: fusion models are not supported");
    if (!(a.at("depths").size() == 4 && a.at("num_attention_heads").size() == 4))
        throw std::runtime_error("CLAP: expected four HTSAT stages");
    if (!(a.at("patch_size") == 4 && a.at("patch_stride") == json::array({4, 4}) &&
          a.at("patch_embed_input_channels") == 1 && a.at("qkv_bias").get<bool>() &&
          a.at("enable_patch_layer_norm").get<bool>() && a.at("flatten_patch_embeds").get<bool>()))
        throw std::runtime_error("CLAP: unsupported patch embedding");
    if (!(a.at("hidden_act") == "gelu" && t.at("hidden_act") == "gelu"))
        throw std::runtime_error("CLAP: only exact GELU is supported");
    if (!(a.at("projection_hidden_act") == "relu" && t.at("projection_hidden_act") == "relu"))
        throw std::runtime_error("CLAP: only ReLU projections supported");
    if (t.value("position_embedding_type", std::string("absolute")) != "absolute")
        throw std::runtime_error("CLAP: unsupported text position embedding type");
    const int spec = a.at("spec_size"), bins = a.at("num_mel_bins"), window = a.at("window_size");
    if (!(spec > 0 && spec % 32 == 0 && bins > 0 && spec % bins == 0 && window > 0 && spec / 32 >= spec / bins))
        throw std::runtime_error("CLAP: invalid spectrogram geometry");
    int dim = a.at("patch_embeds_hidden_size");
    if (!(dim > 0 && dim * 8 == a.at("hidden_size")))
        throw std::runtime_error("CLAP: audio hidden size mismatch");
}

void ClapModel::load_arch_tensors() {
    const int spec = a.at("spec_size"), bins = a.at("num_mel_bins"), window = a.at("window_size");
    int dim = a.at("patch_embeds_hidden_size");
    patch = create_layer(loader, "audio.patch_embed.proj", {4, 4, 1, dim}, {dim});
    patch_norm = create_norm(loader, "audio.patch_embed.norm", dim);
    batch_norm = BatchNormWeights::create(loader, "audio.batch_norm", bins);
    for (int s = 0; s < 4; ++s) {
        const int heads = a.at("num_attention_heads")[s], depth = a.at("depths")[s];
        if (!(heads > 0 && dim % heads == 0 && depth > 0))
            throw std::runtime_error("CLAP: invalid audio attention dimensions");
        if (!(spec / (4 << s) >= window))
            throw std::runtime_error("CLAP: window larger than stage resolution is unsupported");
        Stage stage;
        stage.heads = heads;
        for (int b = 0; b < depth; ++b) {
            const std::string p = "audio.layers." + std::to_string(s) + ".blocks." + std::to_string(b);
            TransformerWeights block;
            block.attention_norm = create_norm(loader, p + ".layernorm_before", dim);
            block.output_norm = create_norm(loader, p + ".layernorm_after", dim);
            block.attention.query = create_linear(loader, p + ".attention.self.query", dim, dim);
            block.attention.key = create_linear(loader, p + ".attention.self.key", dim, dim);
            block.attention.value = create_linear(loader, p + ".attention.self.value", dim, dim);
            block.attention.relative_bias =
                loader.create_tensor(p + ".attention.self.rel_bias", {heads, (2 * window - 1) * (2 * window - 1)});
            block.attention.output = create_linear(loader, p + ".attention.output.dense", dim, dim);
            const int inner = static_cast<int>(dim * a.at("mlp_ratio").get<double>());
            block.intermediate = create_linear(loader, p + ".intermediate.dense", dim, inner);
            block.output = create_linear(loader, p + ".output.dense", inner, dim);
            stage.blocks.push_back(block);
        }
        if (s < 3) {
            const auto p = "audio.layers." + std::to_string(s) + ".downsample";
            stage.norm = create_norm(loader, p + ".norm", dim * 4);
            stage.reduction = create_linear(loader, p + ".reduction", dim * 4, dim * 2, false);
        }
        stages.push_back(std::move(stage));
        dim *= 2;
    }
    audio_norm = create_norm(loader, "audio.norm", dim / 2);
    dim = t.at("hidden_size");
    const int heads = t.at("num_attention_heads"), layers = t.at("num_hidden_layers");
    if (!(dim > 0 && heads > 0 && dim % heads == 0 && layers > 0 && t.at("type_vocab_size").get<int>() > 0))
        throw std::runtime_error("CLAP: invalid text dimensions");
    embeddings = create_embeddings(loader, "text.embeddings", dim, t.at("vocab_size").get<int>(),
                                   t.at("max_position_embeddings").get<int>(), t.at("type_vocab_size").get<int>());
    for (int l = 0; l < layers; ++l)
        text_layers.push_back(create_bert_layer(loader, "text.encoder.layer." + std::to_string(l), dim,
                                                t.at("intermediate_size").get<int>()));
    pooler = create_linear(loader, "text.pooler.dense", dim, dim);
    for (const auto *branch : {"audio", "text"}) {
        const auto &c = std::string(branch) == "audio" ? a : t;
        const int pd = c.at("projection_dim");
        if (!(pd > 0 && pd == config.at("projection_dim")))
            throw std::runtime_error("CLAP: projection dimensions disagree");
        auto &projection = std::string(branch) == "audio" ? audio_projection : text_projection;
        projection.input = create_linear(loader, std::string(branch) + "_proj.linear1", c.at("hidden_size"), pd);
        projection.output = create_linear(loader, std::string(branch) + "_proj.linear2", pd, pd);
    }
    h = {config.at("projection_dim"),
         a.at("num_mel_bins"),
         a.at("spec_size"),
         a.at("patch_embeds_hidden_size"),
         a.at("window_size"),
         t.at("hidden_size"),
         t.at("pad_token_id"),
         t.at("vocab_size"),
         t.at("max_position_embeddings"),
         t.at("num_attention_heads"),
         t.at("bos_token_id"),
         t.at("eos_token_id"),
         a.value("layer_norm_eps", 1e-5f),
         t.at("layer_norm_eps")};
}

TextParameters ClapModel::text_parameters() const { return {h.bos, h.eos, static_cast<size_t>(h.positions - 2)}; }
int ClapModel::projection_dimension() const { return h.projection; }
int ClapModel::mel_bins() const { return h.bins; }
int ClapModel::sample_rate() const { return 48000; }
bool ClapModel::is_waveform_encoder() const { return false; }
AudioFrames ClapModel::audio_frames(const std::vector<float> &) const {
    throw std::runtime_error("CLAP: CLAP exposes one pooled audio projection, not frame representations");
}

ggml_tensor *ClapModel::attention(Graph &g, ggml_tensor *x, const AttentionWeights &weights, int heads, int tokens,
                                  ggml_tensor *mask, int window, int shift, int grid) const {
    auto *ctx = g.ctx();
    ggml_tensor *bias = nullptr;
    if (window) {
        std::vector<int> indices(tokens * tokens);
        for (int i = 0; i < tokens; ++i)
            for (int j = 0; j < tokens; ++j)
                indices[i * tokens + j] =
                    (i / window - j / window + window - 1) * (2 * window - 1) + i % window - j % window + window - 1;
        bias = g.embedding(weights.relative_bias, indices);
        bias = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, bias, heads, tokens, tokens), 2, 0, 1, 3));
        if (shift) {
            const int windows = int(x->ne[1]) / tokens, per_row = grid / window;
            std::vector<float> values(size_t(tokens) * tokens * windows);
            auto region = [&](int r, int c) {
                auto section = [&](int p) { return p < grid - window ? 0 : p < grid - shift ? 1 : 2; };
                return section(r) * 3 + section(c);
            };
            for (int win = 0; win < windows; ++win)
                for (int i = 0; i < tokens; ++i)
                    for (int j = 0; j < tokens; ++j) {
                        const int r = win / per_row * window, c = win % per_row * window;
                        values[(size_t(win) * tokens + i) * tokens + j] =
                            region(r + i / window, c + i % window) == region(r + j / window, c + j % window) ? 0.0f
                                                                                                             : -100.0f;
                    }
            auto *shift_mask =
                ggml_reshape_4d(ctx, g.input(values, tokens, tokens * windows), tokens, tokens, 1, windows);
            // Expand the shared relative bias across windows before adding each shift mask.
            bias = ggml_add(ctx, ggml_repeat_4d(ctx, bias, tokens, tokens, heads, windows), shift_mask);
        }
    } else
        bias = mask;

    return g.attention(g.linear(x, weights.query), g.linear(x, weights.key), g.linear(x, weights.value), heads, bias, 0,
                       tokens);
}

ggml_tensor *ClapModel::project(Graph &g, ggml_tensor *pooled, const Projection &weights) const {
    auto *hidden = ggml_relu(g.ctx(), g.linear(pooled, weights.input));
    return g.l2_normalize(g.linear(hidden, weights.output), 0);
}

std::vector<float> ClapModel::text(const std::vector<int> &ids) const {
    const int dim = h.hidden, pad = h.pad, vocab = h.vocab;
    if (ids.empty() || ids.size() > static_cast<size_t>(h.positions - 2))
        throw std::runtime_error("CLAP: text length outside model context");
    std::vector<int> mask(ids.size()), positions(ids.size());
    int counter = pad;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= vocab)
            throw std::runtime_error("CLAP: token ID out of vocabulary");
        mask[i] = ids[i] != pad;
        positions[i] = mask[i] ? ++counter : pad;
        if (positions[i] >= h.positions)
            throw std::runtime_error("CLAP: text positions overflow");
    }
    if (!std::any_of(mask.begin(), mask.end(), [](int x) { return x; }))
        throw std::runtime_error("CLAP: text contains only padding");
    std::vector<float> scores(mask.size());
    for (size_t i = 0; i < mask.size(); ++i)
        scores[i] = mask[i] ? 0.0f : -1e30f;
    auto outputs = runtime.run_cached(
        "text", {GraphInput(ids), GraphInput(positions), GraphInput(scores, scores.size(), 1)}, [&](Graph &g) {
            auto *ctx = g.ctx();
            auto *x = ggml_add(ctx, g.embedding(embeddings.word, g.dynamic_input(0)),
                               g.embedding(embeddings.position, g.dynamic_input(1)));
            x = ggml_add(ctx, x, g.embedding(embeddings.type, std::vector<int>{0}));
            const float eps = h.text_epsilon;
            x = g.norm(x, embeddings.norm, eps);
            for (const auto &layer : text_layers) {
                auto *y = attention(g, x, layer.attention, h.heads, int(ids.size()), g.dynamic_input(2));
                y = g.linear(y, layer.attention.output);
                x = g.norm(ggml_add(ctx, y, x), layer.attention_norm, eps);
                y = ggml_gelu_erf(ctx, g.linear(x, layer.intermediate));
                y = g.linear(y, layer.output);
                x = g.norm(ggml_add(ctx, y, x), layer.output_norm, eps);
            }
            auto *first = ggml_view_2d(ctx, x, dim, 1, x->nb[1], 0);
            auto *pooled = ggml_tanh(ctx, g.linear(first, pooler));
            return std::vector<ggml_tensor *>{project(g, pooled, text_projection)};
        });
    auto result = std::move(outputs.front().values);
    if (!std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("CLAP: invalid projection norm");
    return result;
}

std::vector<float> ClapModel::waveform_embedding(const std::vector<float> &pcm, size_t crop_offset) const {
    if (mel_bins() != 64)
        throw std::runtime_error("CLAP: waveform frontend requires 64 mel bins");
    return encode_features(clap::log_mel(pcm, 48000, crop_offset), 1001);
}

std::vector<float> ClapModel::encode_features(const std::vector<float> &features, int frames) const {
    const int bins = h.bins, spec = h.spec, ratio = spec / bins;
    const int target_time = spec * ratio;
    if (!(frames > 0 && frames <= target_time && features.size() == static_cast<size_t>(frames) * bins))
        throw std::runtime_error("CLAP: invalid mel feature shape");
    if (!(std::all_of(features.begin(), features.end(), [](float x) { return std::isfinite(x); })))
        throw std::runtime_error("CLAP: non-finite mel features");
    auto outputs = runtime.run_cached("audio", {GraphInput(features, bins, frames)}, [&](Graph &g) {
        auto *ctx = g.ctx();
        auto *mel = g.dynamic_input(0);
        auto *variance = ggml_add(ctx, g.weight(batch_norm.variance), g.input({1e-5f}, 1, 1));
        mel = ggml_div(ctx, ggml_sub(ctx, mel, g.weight(batch_norm.mean)), ggml_sqrt(ctx, variance));
        mel = ggml_add(ctx, ggml_mul(ctx, mel, g.weight(batch_norm.affine.weight)), g.weight(batch_norm.affine.bias));
        ggml_tensor *resized = mel;
        if (frames != target_time) {
            resized = nullptr;
            for (int tap = 0; tap < 4; ++tap) {
                std::vector<int> rows(target_time);
                std::vector<float> coefficients(target_time);
                for (int time = 0; time < target_time; ++time) {
                    const float source = float(time) * (frames - 1) / (target_time - 1);
                    const int base = int(std::floor(source));
                    rows[time] = std::clamp(base + tap - 1, 0, frames - 1);
                    coefficients[time] = cubic(source - base - (tap - 1));
                }
                auto *values = ggml_get_rows(ctx, mel, g.indices(rows));
                values = ggml_mul(ctx, values, g.input(coefficients, 1, target_time));
                resized = resized ? ggml_add(ctx, resized, values) : values;
            }
        }
        const int patchdim = h.patch_width;
        int grid = spec / 4;
        auto *image = ggml_reshape_3d(ctx, resized, bins, spec, ratio);
        image = ggml_cont(ctx, ggml_permute(ctx, image, 1, 0, 2, 3));
        image = ggml_reshape_3d(ctx, image, spec, spec, 1);
        auto *conv = g.weight(patch.weight, false);
        auto *patches = ggml_im2col(ctx, conv, image, 4, 4, 0, 0, 1, 1, true, GGML_TYPE_F32);
        patches = ggml_reshape_2d(ctx, patches, 16, grid * grid);
        auto *x = g.matmul(ggml_reshape_2d(ctx, conv, 16, patchdim), patches);
        x = ggml_add(ctx, x, g.weight(patch.bias));
        x = g.norm(x, patch_norm, 1e-5f);
        const float eps = h.audio_epsilon;
        for (int stage = 0; stage < 4; ++stage) {
            const auto &stage_weights = stages[stage];
            const int depth = int(stage_weights.blocks.size()), heads = stage_weights.heads, window = h.window;
            const int padded = (grid + window - 1) / window * window;
            for (int block = 0; block < depth; ++block) {
                const auto &layer = stage_weights.blocks[block];
                const int shift = grid <= window ? 0 : (block % 2) * (window / 2);
                auto *normalized = g.norm(x, layer.attention_norm, eps);
                std::vector<int> partition(padded * padded, grid * grid), restore(grid * grid);
                for (int r = 0; r < padded; ++r)
                    for (int c = 0; c < padded; ++c) {
                        const int sr = (r + shift) % padded, sc = (c + shift) % padded;
                        const int index = ((r / window) * (padded / window) + c / window) * window * window +
                                          (r % window) * window + c % window;
                        if (sr < grid && sc < grid)
                            partition[index] = sr * grid + sc;
                    }
                for (int r = 0; r < grid; ++r)
                    for (int c = 0; c < grid; ++c) {
                        const int sr = (r - shift + padded) % padded, sc = (c - shift + padded) % padded;
                        restore[r * grid + c] = ((sr / window) * (padded / window) + sc / window) * window * window +
                                                (sr % window) * window + sc % window;
                    }
                auto *windows = ggml_get_rows(ctx, ggml_pad(ctx, normalized, 0, 1, 0, 0), g.indices(partition));
                auto *attended =
                    attention(g, windows, layer.attention, heads, window * window, nullptr, window, shift, padded);
                attended = g.linear(attended, layer.attention.output);
                x = ggml_add(ctx, x, ggml_get_rows(ctx, attended, g.indices(restore)));
                auto *y = ggml_gelu_erf(ctx, g.linear(g.norm(x, layer.output_norm, eps), layer.intermediate));
                x = ggml_add(ctx, x, g.linear(y, layer.output));
            }
            if (stage < 3) {
                if (grid % 2)
                    throw std::runtime_error("CLAP: odd stage grid not supported");
                std::vector<int> rows(grid * grid);
                for (int r = 0; r < grid / 2; ++r)
                    for (int c = 0; c < grid / 2; ++c)
                        for (int part = 0; part < 4; ++part)
                            rows[(r * (grid / 2) + c) * 4 + part] = (2 * r + part % 2) * grid + 2 * c + part / 2;
                auto *merged = ggml_get_rows(ctx, x, g.indices(rows));
                merged = ggml_reshape_2d(ctx, merged, x->ne[0] * 4, grid * grid / 4);
                x = g.linear(g.norm(merged, stage_weights.norm, 1e-5f), stage_weights.reduction);
                grid /= 2;
            }
        }
        x = g.norm(x, audio_norm, 1e-5f);
        return std::vector<ggml_tensor *>{project(g, g.mean_pool(x), audio_projection)};
    });
    auto result = std::move(outputs.front().values);
    if (!std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("CLAP: invalid projection norm");
    return result;
}
} // namespace clap::detail
