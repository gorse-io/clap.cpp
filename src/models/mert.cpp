#include "models.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace clap::detail {
MertModel::MertModel(const ModelLoader &source, int threads, const std::string &tensor_prefix,
                     const std::string &metadata_prefix)
    : loader(source, tensor_prefix, metadata_prefix), runtime(loader, threads, true) {
    load_arch_hparams();
    load_arch_tensors();
    bind_tensors();
}

void MertModel::load_arch_hparams() {
    config = nlohmann::json::parse(loader.metadata("mert.config"));
    if (config.at("model_type") != "mert_model" || config.at("sample_rate") != 24000)
        throw std::runtime_error("expected a 24 kHz MERT-v1 model");
    for (const char *key : {"feature_extractor_cqt", "deepnorm", "do_stable_layer_norm", "feat_proj_layer_norm",
                            "conv_pos_batch_norm", "conv_bias"})
        if ((config.contains(key) && !config.at(key).is_boolean()) ||
            (std::string(key) == "conv_bias" && !config.contains(key)))
            throw std::runtime_error(std::string("invalid MERT boolean ") + key);
    if (config.value("feature_extractor_cqt", false) || config.value("deepnorm", false) ||
        config.at("feat_extract_norm") != "group" || !config.value("feat_proj_layer_norm", true) ||
        config.value("conv_pos_batch_norm", false) || config.at("hidden_act") != "gelu" ||
        config.at("feat_extract_activation") != "gelu")
        throw std::runtime_error("unsupported MERT-v1 frontend/encoder variant");
    auto positive_int = [](const nlohmann::json &value) {
        return value.is_number_integer() && value > 0 && value <= std::numeric_limits<int>::max();
    };
    for (const char *key : {"hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                            "num_conv_pos_embeddings", "num_conv_pos_embedding_groups"})
        if (!positive_int(config.at(key)))
            throw std::runtime_error(std::string("invalid MERT ") + key);
    const int width = config.at("hidden_size"), heads = config.at("num_attention_heads");
    if (width % heads != 0 || width % config.at("num_conv_pos_embedding_groups").get<int>() != 0)
        throw std::runtime_error("invalid MERT attention/position groups");
    const auto &dims = config.at("conv_dim"), &kernels = config.at("conv_kernel"), &strides = config.at("conv_stride");
    if (!dims.is_array() || !kernels.is_array() || !strides.is_array() || dims.empty() ||
        dims.size() != kernels.size() || dims.size() != strides.size())
        throw std::runtime_error("invalid MERT convolution geometry");
    if (config.contains("num_feat_extract_layers") &&
        (!positive_int(config.at("num_feat_extract_layers")) || config.at("num_feat_extract_layers") != dims.size()))
        throw std::runtime_error("invalid MERT convolution geometry: num_feat_extract_layers");
    int64_t receptive = 1, jump = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        for (const auto *array : {&dims, &kernels, &strides})
            if (!positive_int((*array)[i]))
                throw std::runtime_error("invalid MERT convolution dimension");
        receptive += (kernels[i].get<int64_t>() - 1) * jump;
        jump *= strides[i].get<int64_t>();
        if (receptive > 24000 || jump > 24000)
            throw std::runtime_error("unsupported MERT convolution receptive field");
    }
    minimum_samples = static_cast<int>(receptive);
    if (!config.at("layer_norm_eps").is_number())
        throw std::runtime_error("invalid MERT layer_norm_eps");
    const float epsilon = config.at("layer_norm_eps");
    if (!std::isfinite(epsilon) || epsilon <= 0)
        throw std::runtime_error("invalid MERT layer_norm_eps");
}

void MertModel::load_arch_tensors() const {
    int input = 1;
    const auto dims = config.at("conv_dim"), kernels = config.at("conv_kernel");
    for (size_t i = 0; i < dims.size(); ++i) {
        const int output = dims[i];
        const auto prefix = "frontend." + std::to_string(i);
        loader.shape(prefix + ".conv.weight", {kernels[i].get<int>(), input, output});
        if (config.at("conv_bias").get<bool>())
            loader.shape(prefix + ".conv.bias", {output});
        if (i == 0)
            loader.norm_shape(prefix + ".layer_norm", output);
        input = output;
    }
    const int hidden = config.at("hidden_size"), intermediate = config.at("intermediate_size");
    loader.norm_shape("projection.layer_norm", input);
    loader.linear_shape("projection.projection", input, hidden);
    loader.shape("position.weight", {config.at("num_conv_pos_embeddings").get<int>(),
                                     hidden / config.at("num_conv_pos_embedding_groups").get<int>(), hidden});
    loader.shape("position.bias", {hidden});
    loader.norm_shape("encoder.layer_norm", hidden);
    for (int i = 0; i < config.at("num_hidden_layers").get<int>(); ++i) {
        const auto prefix = "layers." + std::to_string(i);
        for (const char *part : {"q_proj", "k_proj", "v_proj", "out_proj"})
            loader.linear_shape(prefix + ".attention." + part, hidden, hidden);
        loader.norm_shape(prefix + ".layer_norm", hidden);
        loader.norm_shape(prefix + ".final_layer_norm", hidden);
        loader.linear_shape(prefix + ".ffn.in", hidden, intermediate);
        loader.linear_shape(prefix + ".ffn.out", intermediate, hidden);
    }
}

void MertModel::bind_tensors() {
    h = {config.at("hidden_size"),
         config.at("num_attention_heads"),
         config.at("num_conv_pos_embeddings"),
         config.at("num_conv_pos_embedding_groups"),
         config.at("layer_norm_eps"),
         config.at("conv_dim"),
         config.at("conv_kernel"),
         config.at("conv_stride"),
         config.at("conv_bias")};
    for (size_t i = 0; i < h.channels.size(); ++i)
        convolutions.push_back(LayerWeights::bind(loader, "frontend." + std::to_string(i) + ".conv", h.conv_bias));
    frontend_norm = LayerWeights::bind(loader, "frontend.0.layer_norm");
    projection_norm = LayerWeights::bind(loader, "projection.layer_norm");
    projection = LayerWeights::bind(loader, "projection.projection");
    position = LayerWeights::bind(loader, "position");
    encoder_norm = LayerWeights::bind(loader, "encoder.layer_norm");
    for (int i = 0; i < config.at("num_hidden_layers").get<int>(); ++i) {
        const auto p = "layers." + std::to_string(i);
        layers.push_back(
            {{LayerWeights::bind(loader, p + ".attention.q_proj"), LayerWeights::bind(loader, p + ".attention.k_proj"),
              LayerWeights::bind(loader, p + ".attention.v_proj"),
              LayerWeights::bind(loader, p + ".attention.out_proj")},
             LayerWeights::bind(loader, p + ".layer_norm"),
             LayerWeights::bind(loader, p + ".final_layer_norm"),
             LayerWeights::bind(loader, p + ".ffn.in"),
             LayerWeights::bind(loader, p + ".ffn.out")});
    }
}

std::vector<float> MertModel::text(const std::vector<int> &) const {
    throw std::runtime_error("MERT-v1 has no text encoder");
}

std::vector<float> MertModel::normalize(const std::vector<float> &pcm) const {
    if (pcm.size() < static_cast<size_t>(minimum_samples) || pcm.size() > (1ULL << 28))
        throw std::runtime_error("MERT waveform is empty, too short or too large");
    double mean = 0, variance = 0;
    for (float v : pcm) {
        if (!std::isfinite(v))
            throw std::runtime_error("non-finite MERT waveform");
        mean += v;
    }
    mean /= pcm.size();
    for (float v : pcm)
        variance += (v - mean) * (v - mean);
    const double scale = 1 / std::sqrt(variance / pcm.size() + 1e-7);
    std::vector<float> result(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i)
        result[i] = static_cast<float>((pcm[i] - mean) * scale);
    return result;
}

// Grouped im2col + GGML matrix multiply, bounded to one group's workspace.
// Stored convolution weights [out,in/group,kernel] are already dense in tap order.
ggml_tensor *MertModel::attention(Graph &g, ggml_tensor *x, const AttentionWeights &weights) const {
    auto *q = g.linear(x, weights.query);
    auto *k = g.linear(x, weights.key);
    auto *v = g.linear(x, weights.value);
    return g.linear(g.attention(q, k, v, h.heads), weights.output);
}

HostMatrix MertModel::forward(const std::vector<float> &pcm, std::vector<float> *layer_mean) const {
    auto outputs =
        runtime.run_cached(layer_mean ? "mean-layers" : "frames", {GraphInput(pcm, 1, pcm.size())}, [&](Graph &g) {
            auto *ctx = g.ctx();
            auto *hidden = g.dynamic_input(0);
            const auto dims = h.channels, kernels = h.kernels, strides = h.strides;
            for (size_t layer = 0; layer < dims.size(); ++layer) {
                hidden = g.conv1d(hidden, convolutions[layer], kernels[layer], strides[layer], 0, 1);
                if (layer == 0) {
                    auto *channels = ggml_cont(ctx, ggml_transpose(ctx, hidden));
                    channels = ggml_norm(ctx, channels, 1e-5f);
                    hidden = ggml_cont(ctx, ggml_transpose(ctx, channels));
                    hidden = ggml_add(ctx, ggml_mul(ctx, hidden, g.weight(frontend_norm.weight)),
                                      g.weight(frontend_norm.bias));
                }
                hidden = ggml_gelu_erf(ctx, hidden);
            }
            const float epsilon = h.epsilon;
            hidden = g.linear(g.norm(hidden, projection_norm, epsilon), projection);
            const int kernel = h.position_kernel;
            auto *position = g.conv1d(hidden, this->position, kernel, 1, kernel / 2, h.position_groups);
            position = ggml_view_2d(ctx, position, hidden->ne[0], hidden->ne[1], position->nb[1], 0);
            const bool stable = config.value("do_stable_layer_norm", false);
            hidden = ggml_add(ctx, hidden, ggml_gelu_erf(ctx, position));
            if (!stable)
                hidden = g.norm(hidden, encoder_norm, epsilon);
            const int layer_count = int(layers.size());
            ggml_tensor *mean = layer_mean ? g.mean_pool(hidden) : nullptr;
            for (int i = 0; i < layer_count; ++i) {
                const auto &layer = layers[i];
                if (stable) {
                    auto *value = attention(g, g.norm(hidden, layer.attention_norm, epsilon), layer.attention);
                    hidden = ggml_add(ctx, value, hidden);
                    value = g.norm(hidden, layer.output_norm, epsilon);
                    value = ggml_gelu_erf(ctx, g.linear(value, layer.intermediate));
                    hidden = ggml_add(ctx, g.linear(value, layer.output), hidden);
                    // Official stable encoder exposes the unnormalized input of each
                    // layer, but replaces the final state with encoder.layer_norm.
                    if (i == layer_count - 1)
                        hidden = g.norm(hidden, encoder_norm, epsilon);
                } else {
                    auto *value = attention(g, hidden, layer.attention);
                    hidden = g.norm(ggml_add(ctx, value, hidden), layer.attention_norm, epsilon);
                    value = ggml_gelu_erf(ctx, g.linear(hidden, layer.intermediate));
                    value = g.linear(value, layer.output);
                    hidden = g.norm(ggml_add(ctx, value, hidden), layer.output_norm, epsilon);
                }
                if (mean)
                    mean = ggml_add(ctx, mean, g.mean_pool(hidden));
            }
            if (mean)
                return std::vector<ggml_tensor *>{hidden, ggml_scale(ctx, mean, 1.0f / (layer_count + 1))};
            return std::vector<ggml_tensor *>{hidden};
        });
    HostMatrix result(int(outputs[0].rows), int(outputs[0].columns));
    result.data = std::move(outputs[0].values);
    if (layer_mean)
        *layer_mean = std::move(outputs[1].values);
    return result;
}

AudioFrames MertModel::audio_frames(const std::vector<float> &pcm) const {
    auto output = forward(normalize(pcm), nullptr);
    return {std::move(output.data), output.rows, output.cols};
}
AudioFrames MertModel::clamp3_features(const std::vector<float> &pcm) const {
    if (pcm.size() < 24000)
        throw std::runtime_error("CLaMP3 MERT extraction requires at least one second of audio");
    auto normalized = normalize(pcm);
    AudioFrames result;
    result.dimension = projection_dimension();
    constexpr size_t window = 5 * 24000;
    for (size_t start = 0; start < normalized.size(); start += window) {
        const auto count = std::min(window, normalized.size() - start);
        if (count < 24000)
            break;
        std::vector<float> chunk(normalized.begin() + start, normalized.begin() + start + count), mean;
        forward(chunk, &mean);
        result.values.insert(result.values.end(), mean.begin(), mean.end());
        ++result.frames;
    }
    return result;
}
} // namespace clap::detail
