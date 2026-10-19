#include "models.h"
#include <ggml-cpu.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace clap::detail {
using json = nlohmann::json;
namespace {
// PyTorch bicubic interpolation, A=-0.75, align_corners=True.
float cubic(float x) {
    x = std::abs(x);
    if (x <= 1) return ((1.25f * x - 2.25f) * x) * x + 1;
    if (x < 2) return ((-0.75f * x + 3.75f) * x - 6) * x + 3;
    return 0;
}
} // namespace

ClapModel::ClapModel(const ModelLoader &source, int threads) : loader(source), ops(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
}

void ClapModel::load_arch_hparams() {
    config = json::parse(loader.metadata("clap.config"));
    a = config.at("audio_config");
    t = config.at("text_config");

    check(config.at("model_type") == "clap", "not a CLAP model");
    check(!a.at("enable_fusion").get<bool>(), "fusion models are not supported");
    check(a.at("depths").size() == 4 && a.at("num_attention_heads").size() == 4, "expected four HTSAT stages");
    check(a.at("patch_size") == 4 && a.at("patch_stride") == json::array({4, 4}) &&
          a.at("patch_embed_input_channels") == 1 && a.at("qkv_bias").get<bool>() &&
          a.at("enable_patch_layer_norm").get<bool>() && a.at("flatten_patch_embeds").get<bool>(), "unsupported patch embedding");
    check(a.at("hidden_act") == "gelu" && t.at("hidden_act") == "gelu", "only exact GELU is supported");
    check(a.at("projection_hidden_act") == "relu" && t.at("projection_hidden_act") == "relu", "only ReLU projections supported");
    check(t.value("position_embedding_type", std::string("absolute")) == "absolute", "unsupported text position embedding type");
    const int spec = a.at("spec_size"), bins = a.at("num_mel_bins"), window = a.at("window_size");
    check(spec > 0 && spec % 32 == 0 && bins > 0 && spec % bins == 0 && window > 0 &&
          spec / 32 >= spec / bins, "invalid spectrogram geometry");
    int dim = a.at("patch_embeds_hidden_size");
    check(dim > 0 && dim * 8 == a.at("hidden_size"), "audio hidden size mismatch");
}

void ClapModel::load_arch_tensors() const {
    const int spec = a.at("spec_size"), bins = a.at("num_mel_bins"), window = a.at("window_size");
    int dim = a.at("patch_embeds_hidden_size");
    loader.shape("audio.patch_embed.proj.weight", {4, 4, 1, dim});
    loader.shape("audio.patch_embed.proj.bias", {dim});
    loader.norm_shape("audio.patch_embed.norm", dim);
    loader.norm_shape("audio.batch_norm", bins);
    loader.shape("audio.batch_norm.running_mean", {bins}); loader.shape("audio.batch_norm.running_var", {bins});
    for (int s = 0; s < 4; ++s) {
        const int heads = a.at("num_attention_heads")[s], depth = a.at("depths")[s];
        check(heads > 0 && dim % heads == 0 && depth > 0, "invalid audio attention dimensions");
        check(spec / (4 << s) >= window, "window larger than stage resolution is unsupported");
        for (int b = 0; b < depth; ++b) {
            const std::string p = "audio.layers." + std::to_string(s) + ".blocks." + std::to_string(b);
            loader.norm_shape(p + ".layernorm_before", dim); loader.norm_shape(p + ".layernorm_after", dim);
            for (const auto *q : {"query", "key", "value"}) loader.linear_shape(p + ".attention.self." + q, dim, dim);
            loader.shape(p + ".attention.self.rel_bias", {heads, (2 * window - 1) * (2 * window - 1)});
            loader.linear_shape(p + ".attention.output.dense", dim, dim);
            const int inner = static_cast<int>(dim * a.at("mlp_ratio").get<double>());
            loader.linear_shape(p + ".intermediate.dense", dim, inner); loader.linear_shape(p + ".output.dense", inner, dim);
        }
        if (s < 3) {
            const auto p = "audio.layers." + std::to_string(s) + ".downsample";
            loader.norm_shape(p + ".norm", dim * 4); loader.linear_shape(p + ".reduction", dim * 4, dim * 2, false);
        }
        dim *= 2;
    }
    loader.norm_shape("audio.norm", dim / 2);
    dim = t.at("hidden_size");
    const int heads = t.at("num_attention_heads"), layers = t.at("num_hidden_layers");
    check(dim > 0 && heads > 0 && dim % heads == 0 && layers > 0 &&
          t.at("type_vocab_size").get<int>() > 0, "invalid text dimensions");
    loader.shape("text.embeddings.word_embeddings.weight", {dim, t.at("vocab_size").get<int>()});
    loader.shape("text.embeddings.position_embeddings.weight", {dim, t.at("max_position_embeddings").get<int>()});
    loader.shape("text.embeddings.token_type_embeddings.weight", {dim, t.at("type_vocab_size").get<int>()});
    loader.norm_shape("text.embeddings.LayerNorm", dim);
    for (int l = 0; l < layers; ++l) {
        const auto p = "text.encoder.layer." + std::to_string(l);
        for (const auto *q : {"query", "key", "value"}) loader.linear_shape(p + ".attention.self." + q, dim, dim);
        loader.linear_shape(p + ".attention.output.dense", dim, dim); loader.norm_shape(p + ".attention.output.LayerNorm", dim);
        const int inner = t.at("intermediate_size");
        loader.linear_shape(p + ".intermediate.dense", dim, inner); loader.linear_shape(p + ".output.dense", inner, dim);
        loader.norm_shape(p + ".output.LayerNorm", dim);
    }
    loader.linear_shape("text.pooler.dense", dim, dim);
    for (const auto *branch : {"audio", "text"}) {
        const auto &c = std::string(branch) == "audio" ? a : t;
        const int pd = c.at("projection_dim");
        check(pd > 0 && pd == config.at("projection_dim"), "projection dimensions disagree");
        loader.linear_shape(std::string(branch) + "_proj.linear1", c.at("hidden_size"), pd);
        loader.linear_shape(std::string(branch) + "_proj.linear2", pd, pd);
    }
}

TextParameters ClapModel::text_parameters() const {
    return {t.at("bos_token_id"), t.at("eos_token_id"),
            static_cast<size_t>(t.at("max_position_embeddings").get<int>() - 2)};
}
int ClapModel::projection_dimension() const { return config.at("projection_dim"); }
int ClapModel::mel_bins() const { return a.at("num_mel_bins"); }
int ClapModel::sample_rate() const { return 48000; }
bool ClapModel::is_waveform_encoder() const { return false; }
AudioFrames ClapModel::audio_frames(const std::vector<float> &) const {
    check(false, "CLAP exposes one pooled audio projection, not frame representations");
    return {};
}

Matrix ClapModel::attention(const Matrix &x, const std::string &prefix, int heads, int tokens,
                 const std::vector<int> *mask, int window, int shift, int grid) const {
    const auto q = ops.linear(x, prefix + ".query"), k = ops.linear(x, prefix + ".key"), v = ops.linear(x, prefix + ".value");
    Matrix result(x.rows, x.cols);
    const int d = x.cols / heads;
    const float scale = 1 / std::sqrt(static_cast<float>(d));
    const auto *bias = window ? loader.weight(prefix + ".rel_bias") : nullptr;
    std::vector<float> scores(tokens);
    auto region = [&](int r, int c) {
        const auto section = [&](int p) { return p < grid - window ? 0 : p < grid - shift ? 1 : 2; };
        return section(r) * 3 + section(c);
    };
    for (int start = 0; start < x.rows; start += tokens) {
        const int win = start / tokens, windows_per_row = window ? grid / window : 0;
        for (int h = 0; h < heads; ++h) for (int i = 0; i < tokens; ++i) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < tokens; ++j) {
                float score = 0;
                for (int z = 0; z < d; ++z) score += q[start + i][h * d + z] * k[start + j][h * d + z];
                score *= scale;
                if (mask && (*mask)[j] == 0) score = -1e30f;
                if (window) {
                    const int pos = (i / window - j / window + window - 1) * (2 * window - 1) + i % window - j % window + window - 1;
                    score += scalar(bias, static_cast<size_t>(pos) * heads + h);
                    if (shift) {
                        const int r = win / windows_per_row * window, c = win % windows_per_row * window;
                        if (region(r + i / window, c + i % window) != region(r + j / window, c + j % window)) score -= 100;
                    }
                }
                scores[j] = score; maximum = std::max(maximum, score);
            }
            float sum = 0;
            for (float &s : scores) { s = std::exp(s - maximum); sum += s; }
            for (int j = 0; j < tokens; ++j) {
                const float p = scores[j] / sum;
                for (int z = 0; z < d; ++z) result[start + i][h * d + z] += p * v[start + j][h * d + z];
            }
        }
    }
    return result;
}

std::vector<float> ClapModel::project(const Matrix &pooled, const std::string &prefix) const {
    auto h = ops.linear(pooled, prefix + ".linear1");
    for (float &v : h.data) v = std::max(v, 0.0f);
    auto y = ops.linear(h, prefix + ".linear2");
    double square = 0;
    for (float v : y.data) square += v * v;
    check(std::isfinite(square) && square > 0, "invalid projection norm");
    const float length = static_cast<float>(std::sqrt(square));
    for (float &v : y.data) v /= length;
    return y.data;
}

std::vector<float> ClapModel::text(const std::vector<int> &ids) const {
    const int dim = t.at("hidden_size"), pad = t.at("pad_token_id"), vocab = t.at("vocab_size");
    check(!ids.empty() && ids.size() <= static_cast<size_t>(t.at("max_position_embeddings").get<int>() - 2), "text length outside model context");
    const auto *word = loader.weight("text.embeddings.word_embeddings.weight"), *position = loader.weight("text.embeddings.position_embeddings.weight"),
               *type = loader.weight("text.embeddings.token_type_embeddings.weight");
    Matrix x(static_cast<int>(ids.size()), dim);
    std::vector<int> mask(ids.size());
    int counter = pad;
    for (int i = 0; i < x.rows; ++i) {
        check(ids[i] >= 0 && ids[i] < vocab, "token ID out of vocabulary");
        mask[i] = ids[i] != pad;
        const int pos = mask[i] ? ++counter : pad;
        check(pos < t.at("max_position_embeddings").get<int>(), "text positions overflow");
        for (int c = 0; c < dim; ++c) x[i][c] = scalar(word, static_cast<size_t>(ids[i]) * dim + c) + scalar(position, static_cast<size_t>(pos) * dim + c) + scalar(type, c);
    }
    check(std::any_of(mask.begin(), mask.end(), [](int x) { return x; }), "text contains only padding");
    const float eps = t.at("layer_norm_eps");
    x = ops.norm(x, "text.embeddings.LayerNorm", eps);
    for (int l = 0; l < t.at("num_hidden_layers").get<int>(); ++l) {
        const auto p = "text.encoder.layer." + std::to_string(l);
        auto y = attention(x, p + ".attention.self", t.at("num_attention_heads"), x.rows, &mask);
        y = ops.linear(y, p + ".attention.output.dense"); add(y, x);
        x = ops.norm(y, p + ".attention.output.LayerNorm", eps);
        y = ops.linear(x, p + ".intermediate.dense"); gelu(y);
        y = ops.linear(y, p + ".output.dense"); add(y, x);
        x = ops.norm(y, p + ".output.LayerNorm", eps);
    }
    Matrix first(1, dim); std::copy_n(x[0], dim, first[0]);
    auto pooled = ops.linear(first, "text.pooler.dense");
    for (float &v : pooled.data) v = std::tanh(v);
    return project(pooled, "text_proj");
}

std::vector<float> ClapModel::audio_features(const std::vector<float> &features, int frames) const {
    const int bins = a.at("num_mel_bins"), spec = a.at("spec_size"), ratio = spec / bins;
    const int target_time = spec * ratio;
    check(frames > 0 && frames <= target_time && features.size() == static_cast<size_t>(frames) * bins, "invalid mel feature shape");
    check(std::all_of(features.begin(), features.end(), [](float x) { return std::isfinite(x); }), "non-finite mel features");
    Matrix mel(frames, bins);
    const auto *bnw = loader.weight("audio.batch_norm.weight"), *bnb = loader.weight("audio.batch_norm.bias"),
               *mean = loader.weight("audio.batch_norm.running_mean"), *variance = loader.weight("audio.batch_norm.running_var");
    for (int t = 0; t < frames; ++t) for (int f = 0; f < bins; ++f)
        mel[t][f] = (features[static_cast<size_t>(t) * bins + f] - scalar(mean, f)) / std::sqrt(scalar(variance, f) + 1e-5f) * scalar(bnw, f) + scalar(bnb, f);
    Matrix resized(target_time, bins);
    if (frames == target_time) resized = std::move(mel);
    else for (int t = 0; t < target_time; ++t) {
        const float source = static_cast<float>(t) * (frames - 1) / (target_time - 1);
        const int base = static_cast<int>(std::floor(source));
        const float frac = source - base;
        float coeff[4]; for (int k = 0; k < 4; ++k) coeff[k] = cubic(frac - (k - 1));
        for (int f = 0; f < bins; ++f) for (int k = 0; k < 4; ++k)
            resized[t][f] += mel[std::max(0, std::min(frames - 1, base + k - 1))][f] * coeff[k];
    }
    // mel2img: [ratio, time/ratio, bins] -> [ratio, bins, time/ratio]
    // -> [spec,spec]. Patch convolution lowered to one ggml matrix multiply.
    const int patchdim = a.at("patch_embeds_hidden_size");
    int grid = spec / 4;
    Matrix patches(grid * grid, 16);
    for (int r = 0; r < grid; ++r) for (int c = 0; c < grid; ++c)
        for (int y = 0; y < 4; ++y) for (int z = 0; z < 4; ++z) {
            const int image_row = r * 4 + y, image_col = c * 4 + z;
            const int time = (image_row / bins) * spec + image_col, freq = image_row % bins;
            patches[r * grid + c][y * 4 + z] = resized[time][freq];
        }
    // The stored Conv2d [out,1,4,4] has the same contiguous bytes as [out,16].
    const auto *conv = loader.weight("audio.patch_embed.proj.weight");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> convctx(ggml_init({4096, nullptr, true}), ggml_free);
    check(convctx != nullptr, "conv metadata allocation failed");
    auto *flat = ggml_new_tensor_2d(convctx.get(), conv->type, 16, patchdim);
    flat->data = conv->data;
    // Use a temporary alias name only in this instance's metadata context.
    // Avoid changing the immutable stored tensor: linear accepts a matrix view.
    auto conv_linear = [&](const Matrix &input) {
        const size_t bytes = (input.data.size() + static_cast<size_t>(input.rows) * patchdim) * sizeof(float) + 2 * 1024 * 1024;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
        check(ctx != nullptr, "patch allocation failed");
        auto *in = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, input.rows);
        std::memcpy(in->data, input.data.data(), input.data.size() * sizeof(float));
        auto *out = ggml_mul_mat(ctx.get(), flat, in);
        auto *graph = ggml_new_graph_custom(ctx.get(), 32, false); ggml_build_forward_expand(graph, out);
        check(ggml_graph_compute_with_ctx(ctx.get(), graph, ops.threads()) == GGML_STATUS_SUCCESS, "patch convolution failed");
        Matrix result(input.rows, patchdim);
        std::memcpy(result.data.data(), out->data, result.data.size() * sizeof(float));
        const auto *bias = loader.weight("audio.patch_embed.proj.bias");
        for (int r = 0; r < result.rows; ++r) for (int c = 0; c < patchdim; ++c) result[r][c] += scalar(bias, c);
        return result;
    };
    Matrix x = ops.norm(conv_linear(patches), "audio.patch_embed.norm", 1e-5f);
    const float eps = a.at("layer_norm_eps");
    for (int stage = 0; stage < 4; ++stage) {
        const int depth = a.at("depths")[stage], heads = a.at("num_attention_heads")[stage], window = a.at("window_size");
        const int padded = (grid + window - 1) / window * window;
        for (int block = 0; block < depth; ++block) {
            const auto p = "audio.layers." + std::to_string(stage) + ".blocks." + std::to_string(block);
            const int shift = grid <= window ? 0 : (block % 2) * (window / 2);
            auto normalized = ops.norm(x, p + ".layernorm_before", eps);
            Matrix windows(padded * padded, x.cols);
            // Partition a cyclically shifted, zero-padded image into windows.
            for (int r = 0; r < padded; ++r) for (int c = 0; c < padded; ++c) {
                const int sr = (r + shift) % padded, sc = (c + shift) % padded;
                const int index = ((r / window) * (padded / window) + c / window) * window * window + (r % window) * window + c % window;
                if (sr < grid && sc < grid) std::copy_n(normalized[sr * grid + sc], x.cols, windows[index]);
            }
            auto attended = attention(windows, p + ".attention.self", heads, window * window, nullptr, window, shift, padded);
            attended = ops.linear(attended, p + ".attention.output.dense");
            Matrix y(x.rows, x.cols);
            for (int r = 0; r < grid; ++r) for (int c = 0; c < grid; ++c) {
                const int sr = (r - shift + padded) % padded, sc = (c - shift + padded) % padded;
                const int index = ((sr / window) * (padded / window) + sc / window) * window * window + (sr % window) * window + sc % window;
                std::copy_n(attended[index], x.cols, y[r * grid + c]);
            }
            add(x, y);
            y = ops.linear(ops.norm(x, p + ".layernorm_after", eps), p + ".intermediate.dense"); gelu(y);
            y = ops.linear(y, p + ".output.dense"); add(x, y);
        }
        if (stage < 3) {
            check(grid % 2 == 0, "odd stage grid not supported");
            Matrix merged((grid / 2) * (grid / 2), x.cols * 4);
            for (int r = 0; r < grid / 2; ++r) for (int c = 0; c < grid / 2; ++c)
                for (int part = 0; part < 4; ++part) {
                    const int sr = 2 * r + part % 2, sc = 2 * c + part / 2;
                    std::copy_n(x[sr * grid + sc], x.cols, merged[r * (grid / 2) + c] + part * x.cols);
                }
            const auto p = "audio.layers." + std::to_string(stage) + ".downsample";
            x = ops.linear(ops.norm(merged, p + ".norm", 1e-5f), p + ".reduction", false);
            grid /= 2;
        }
    }
    x = ops.norm(x, "audio.norm", 1e-5f);
    Matrix pooled(1, x.cols);
    for (int r = 0; r < x.rows; ++r) for (int c = 0; c < x.cols; ++c) pooled[0][c] += x[r][c] / x.rows;
    return project(pooled, "audio_proj");
}} // namespace clap::detail
