#include "audio.h"
#include "models.h"
#include "quant.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace clap::detail {
MuqModel::MuqModel(const ModelLoader &source, int threads) : loader(source), runtime(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
    bind_tensors();
}
void MuqModel::load_arch_hparams() {
    config = nlohmann::json::parse(loader.metadata("muq.config"));
    encoder = config.at("w2v2_config");
    // Validate exact JSON integer types and wide bounds before ANY int/float
    // narrowing. This is also the MuLan dependency gate during construction.
    auto dimension = [](const nlohmann::json &obj, const char *key, int64_t maximum = 65536) -> int64_t {
        const auto &value = obj.at(key);
        // Compare in JSON's numeric domain before extracting signed int64_t,
        // so uint64 values cannot wrap through a signed conversion.
        if (!value.is_number_integer() || value <= 0 || value > maximum)
            throw std::runtime_error(std::string("CLAP: invalid MuQ dimension: ") + key);
        return value.get<int64_t>();
    };
    for (const auto &field : {std::pair<const char *, int>{"n_mels", 128}, {"hop_length", 240}, {"label_rate", 25}})
        if (dimension(config, field.first) != field.second)
            throw std::runtime_error(std::string("CLAP: invalid MuQ ") + field.first);
    if (config.at("features") != nlohmann::json::array({"melspec_2048"}) || config.value("is_flash", false))
        throw std::runtime_error("CLAP: unsupported MuQ frontend/variant");
    const int64_t hidden = dimension(config, "encoder_dim"), heads = dimension(encoder, "num_attention_heads");
    const int64_t channels = dimension(config, "conv_dim"), depth = dimension(config, "encoder_depth", 256);
    const int64_t intermediate = dimension(encoder, "intermediate_size");
    const int64_t kernel = dimension(encoder, "conv_depthwise_kernel_size", 4095);
    if (hidden % heads || (hidden / heads) % 2 || kernel % 2 != 1)
        throw std::runtime_error("CLAP: invalid MuQ attention/kernel geometry");
    // All operands are bounded int64_t above. Even the largest budget fits
    // int64_t; the signed-int ceiling bounds tensor products and schema cost.
    // Keep these limits in sync with conversion/muq.py. Unused HF/RVQ fields
    // (notably HF num_hidden_layers) do not define this inference backbone.
    const int64_t layer =
        4 * hidden * intermediate + 7 * hidden * hidden + hidden * kernel + 64 * hidden + 4 * intermediate;
    const int64_t budget = 54 * channels * channels + 32 * channels * hidden + depth * layer;
    if (budget > std::numeric_limits<int>::max())
        throw std::runtime_error("CLAP: invalid MuQ derived tensor products");
    if (!(encoder.at("position_embeddings_type") == "rotary" &&
          (encoder.at("hidden_act") == "swish" || encoder.at("hidden_act") == "silu")))
        throw std::runtime_error("CLAP: MuQ requires rotary/swish encoder");
    auto scalar = [](const nlohmann::json &obj, const char *key, bool positive) {
        const auto &value = obj.at(key);
        if (!value.is_number())
            throw std::runtime_error(std::string("CLAP: invalid MuQ scalar: ") + key);
        const double wide = value.get<double>();
        if (!std::isfinite(wide) || std::abs(wide) > std::numeric_limits<float>::max() ||
            (wide != 0 && std::abs(wide) < std::numeric_limits<float>::denorm_min()) || (positive && wide <= 0))
            throw std::runtime_error(std::string("CLAP: invalid MuQ scalar: ") + key);
    };
    scalar(encoder, "rotary_embedding_base", true);
    scalar(encoder, "layer_norm_eps", true);
    scalar(config.at("stat"), "melspec_2048_std", true);
    scalar(config.at("stat"), "melspec_2048_mean", false);
}
void MuqModel::load_arch_tensors() const {
    loader.shape("frontend.window", {2048});
    loader.shape("frontend.filters", {128, 1025});
    const int channels = config.at("conv_dim"), hidden = config.at("encoder_dim");
    auto batch = [&](const std::string &name, int width) {
        loader.norm_shape(name, width);
        loader.shape(name + ".running_mean", {width});
        loader.shape(name + ".running_var", {width});
    };
    for (int stage = 0; stage < 2; ++stage) {
        const std::string root = "conv.conv." + std::to_string(stage);
        for (int layer = 1; layer <= 3; ++layer) {
            const int input = stage == 0 && layer != 2 ? 1 : channels;
            const auto conv = root + ".conv" + std::to_string(layer);
            loader.shape(conv + ".weight", {3, 3, input, channels});
            loader.shape(conv + ".bias", {channels});
            batch(root + ".bn" + std::to_string(layer), channels);
        }
    }
    loader.linear_shape("conv.linear", channels * 32, hidden);
    loader.norm_shape("conformer.layer_norm", hidden);
    for (int layer = 0; layer < config.at("encoder_depth").get<int>(); ++layer) {
        const std::string root = "conformer.layers." + std::to_string(layer);
        for (const auto *norm : {"ffn1_layer_norm", "ffn2_layer_norm", "self_attn_layer_norm", "final_layer_norm"})
            loader.norm_shape(root + "." + norm, hidden);
        for (const auto *ffn : {"ffn1", "ffn2"}) {
            loader.linear_shape(root + "." + ffn + ".intermediate_dense", hidden, encoder.at("intermediate_size"));
            loader.linear_shape(root + "." + ffn + ".output_dense", encoder.at("intermediate_size"), hidden);
        }
        for (const auto *projection : {"linear_q", "linear_k", "linear_v", "linear_out"})
            loader.linear_shape(root + ".self_attn." + projection, hidden, hidden);
        loader.norm_shape(root + ".conv_module.layer_norm", hidden);
        loader.shape(root + ".conv_module.pointwise_conv1.weight", {1, hidden, hidden * 2});
        loader.shape(root + ".conv_module.depthwise_conv.weight",
                     {encoder.at("conv_depthwise_kernel_size").get<int>(), 1, hidden});
        batch(root + ".conv_module.batch_norm", hidden);
        loader.shape(root + ".conv_module.pointwise_conv2.weight", {1, hidden, hidden});
    }
}

void MuqModel::bind_tensors() {
    h = {config.at("encoder_dim"),
         encoder.at("num_attention_heads"),
         encoder.at("conv_depthwise_kernel_size"),
         encoder.at("layer_norm_eps"),
         encoder.at("rotary_embedding_base"),
         config.at("stat").at("melspec_2048_mean"),
         config.at("stat").at("melspec_2048_std")};
    window = loader.weight("frontend.window");
    filters = loader.weight("frontend.filters");
    for (int i = 0; i < 2; ++i) {
        const auto p = "conv.conv." + std::to_string(i);
        stages.push_back({LayerWeights::bind(loader, p + ".conv1"), LayerWeights::bind(loader, p + ".conv2"),
                          LayerWeights::bind(loader, p + ".conv3"), BatchNormWeights::bind(loader, p + ".bn1"),
                          BatchNormWeights::bind(loader, p + ".bn2"), BatchNormWeights::bind(loader, p + ".bn3")});
    }
    projection = LayerWeights::bind(loader, "conv.linear");
    final_norm = LayerWeights::bind(loader, "conformer.layer_norm");
    for (int i = 0; i < config.at("encoder_depth").get<int>(); ++i) {
        const auto p = "conformer.layers." + std::to_string(i);
        ConformerWeights layer;
        layer.ffn1 = {LayerWeights::bind(loader, p + ".ffn1_layer_norm"),
                      LayerWeights::bind(loader, p + ".ffn1.intermediate_dense"),
                      LayerWeights::bind(loader, p + ".ffn1.output_dense")};
        layer.ffn2 = {LayerWeights::bind(loader, p + ".ffn2_layer_norm"),
                      LayerWeights::bind(loader, p + ".ffn2.intermediate_dense"),
                      LayerWeights::bind(loader, p + ".ffn2.output_dense")};
        layer.attention_norm = LayerWeights::bind(loader, p + ".self_attn_layer_norm");
        layer.final_norm = LayerWeights::bind(loader, p + ".final_layer_norm");
        layer.attention = {LayerWeights::bind(loader, p + ".self_attn.linear_q"),
                           LayerWeights::bind(loader, p + ".self_attn.linear_k"),
                           LayerWeights::bind(loader, p + ".self_attn.linear_v"),
                           LayerWeights::bind(loader, p + ".self_attn.linear_out")};
        layer.convolution.norm = LayerWeights::bind(loader, p + ".conv_module.layer_norm");
        layer.convolution.input = LayerWeights::bind(loader, p + ".conv_module.pointwise_conv1", false);
        layer.convolution.depthwise = LayerWeights::bind(loader, p + ".conv_module.depthwise_conv", false);
        layer.convolution.batch_norm = BatchNormWeights::bind(loader, p + ".conv_module.batch_norm");
        layer.convolution.output = LayerWeights::bind(loader, p + ".conv_module.pointwise_conv2", false);
        layers.push_back(layer);
    }
}

TextParameters MuqModel::text_parameters() const { return {0, 0, 0}; }
int MuqModel::projection_dimension() const { return h.hidden; }
int MuqModel::mel_bins() const { return 128; }
int MuqModel::sample_rate() const { return 24000; }
bool MuqModel::is_waveform_encoder() const { return true; }
std::vector<float> MuqModel::text(const std::vector<int> &) const {
    throw std::runtime_error("CLAP: MuQ has no text encoder");
}
HostMatrix MuqModel::frontend(const std::vector<float> &pcm) const {
    const int frames = static_cast<int>(pcm.size() / 240);
    HostMatrix mel(frames, 128);
    const float mean = h.mean, stddev = h.stddev;
    std::vector<std::complex<double>> spectrum(2048);
    std::vector<float> power(1025);
    for (int frame = 0; frame < frames; ++frame) {
        for (int i = 0; i < 2048; ++i) {
            int source = frame * 240 + i - 1024;
            if (source < 0)
                source = -source;
            if (source >= static_cast<int>(pcm.size()))
                source = 2 * static_cast<int>(pcm.size()) - 2 - source;
            spectrum[i] = pcm[source] * scalar(window, i);
        }
        clap::fft(spectrum);
        for (int i = 0; i < 1025; ++i)
            power[i] = std::norm(spectrum[i]);
        for (int bin = 0; bin < 128; ++bin) {
            float value = 0;
            for (int frequency = 0; frequency < 1025; ++frequency)
                value += power[frequency] * scalar(filters, static_cast<size_t>(frequency) * 128 + bin);
            mel[frame][bin] = (10.0f * std::log10(std::max(value, 1e-10f)) - mean) / stddev;
        }
    }
    return mel;
}
ggml_tensor *MuqModel::batch_norm(Graph &g, ggml_tensor *x, const BatchNormWeights &weights) const {
    auto *ctx = g.ctx();
    auto *variance = ggml_add(ctx, g.weight(weights.variance), g.input({1e-5f}, 1, 1));
    auto *scale = ggml_div(ctx, g.weight(weights.affine.weight), ggml_sqrt(ctx, variance));
    return ggml_add(ctx, ggml_mul(ctx, ggml_sub(ctx, x, g.weight(weights.mean)), scale), g.weight(weights.affine.bias));
}
MuqModel::Image MuqModel::conv2d(Graph &g, const Image &input, const LayerWeights &weights, int stride) const {
    auto *ctx = g.ctx();
    auto *weight = g.weight(weights.weight);
    const int frequencies = (input.frequencies + stride - 1) / stride, frames = (input.frames + stride - 1) / stride;
    const int columns = int(weight->ne[3]), channels = int(input.values->ne[0]);
    auto *data = ggml_reshape_3d(ctx, input.values, channels, input.frames, input.frequencies);
    data = ggml_cont(ctx, ggml_permute(ctx, data, 2, 0, 1, 3));
    data = ggml_pad_ext(ctx, data, 1, 1, 0, 0, 0, 0, 0, 0);
    auto *flat = ggml_reshape_2d(ctx, weight, channels * 9, columns);
    std::vector<ggml_tensor *> tiles;
    // Bound im2col scratch for long songs; concatenate time tiles in a balanced tree.
    for (int start = 0; start < frames; start += 32) {
        const int count = std::min(32, frames - start), span = (count - 1) * stride + 3;
        auto *tile = ggml_view_3d(ctx, data, span, input.frequencies, channels, data->nb[1], data->nb[2],
                                  size_t(start * stride) * data->nb[0]);
        auto *patches = ggml_im2col(ctx, weight, ggml_cont(ctx, tile), stride, stride, 0, 1, 1, 1, true, GGML_TYPE_F32);
        patches = ggml_reshape_2d(ctx, patches, channels * 9, count * frequencies);
        auto *output = g.matmul(flat, patches);
        tiles.push_back(ggml_reshape_3d(ctx, output, columns, count, frequencies));
    }
    while (tiles.size() > 1) {
        std::vector<ggml_tensor *> merged;
        for (size_t i = 0; i < tiles.size(); i += 2)
            merged.push_back(i + 1 < tiles.size() ? ggml_concat(ctx, tiles[i], tiles[i + 1], 1) : tiles[i]);
        tiles = std::move(merged);
    }
    auto *result = ggml_reshape_2d(ctx, tiles.front(), columns, frames * frequencies);
    result = ggml_add(ctx, result, g.weight(weights.bias));
    return {frequencies, frames, result};
}
ggml_tensor *MuqModel::subsample(Graph &g, ggml_tensor *mel) const {
    auto *ctx = g.ctx();
    auto *data = ggml_cont(ctx, ggml_transpose(ctx, mel));
    Image value{128, int(mel->ne[1]), ggml_reshape_2d(ctx, data, 1, ggml_nelements(data))};
    for (const auto &stage : stages) {
        auto residual = conv2d(g, value, stage.conv3, 2);
        residual.values = batch_norm(g, residual.values, stage.bn3);
        value = conv2d(g, value, stage.conv1, 2);
        value.values = ggml_relu(ctx, batch_norm(g, value.values, stage.bn1));
        value = conv2d(g, value, stage.conv2, 1);
        value.values = batch_norm(g, value.values, stage.bn2);
        value.values = ggml_relu(ctx, ggml_add(ctx, value.values, residual.values));
    }
    data = ggml_reshape_3d(ctx, value.values, value.values->ne[0], value.frames, value.frequencies);
    data = ggml_cont(ctx, ggml_permute(ctx, data, 1, 2, 0, 3));
    return g.linear(ggml_reshape_2d(ctx, data, value.values->ne[0] * value.frequencies, value.frames), projection);
}
ggml_tensor *MuqModel::attention(Graph &g, ggml_tensor *hidden, const AttentionWeights &weights) const {
    const int heads = h.heads;
    // Wav2Vec2-Conformer rotates hidden states before Q/K projections.
    auto *rotated = g.rope(hidden, heads, h.rotary_base);
    auto *query = g.linear(rotated, weights.query), *key = g.linear(rotated, weights.key);
    auto *value = g.linear(hidden, weights.value);
    return g.linear(g.attention(query, key, value, heads), weights.output);
}
ggml_tensor *MuqModel::conformer_convolution(Graph &g, ggml_tensor *hidden, const ConvolutionWeights &weights) const {
    auto *ctx = g.ctx();
    const int width = int(hidden->ne[0]), kernel = h.kernel;
    auto *value = g.norm(hidden, weights.norm, 1e-5f);
    value = g.conv1d(value, weights.input, 1, 1, 0, 1);
    auto *gated = ggml_mul(ctx, g.slice(value, 0, width), ggml_sigmoid(ctx, g.slice(value, width, width)));
    value = g.conv1d(gated, weights.depthwise, kernel, 1, (kernel - 1) / 2, width);
    value = ggml_silu(ctx, batch_norm(g, value, weights.batch_norm));
    return g.conv1d(value, weights.output, 1, 1, 0, 1);
}
AudioFrames MuqModel::audio_frames(const std::vector<float> &pcm) const {
    if (!(pcm.size() >= 1025))
        throw std::runtime_error("CLAP: MuQ waveform must contain at least 1025 samples");
    if (!(pcm.size() <= (size_t{1} << 30) / sizeof(float)))
        throw std::runtime_error("CLAP: MuQ waveform exceeds 1 GiB");
    if (!(std::all_of(pcm.begin(), pcm.end(), [](float v) { return std::isfinite(v); })))
        throw std::runtime_error("CLAP: MuQ waveform contains non-finite samples");
    const auto mel = frontend(pcm);
    auto outputs = runtime.run_cached(
        "frames", {GraphInput(mel)},
        [&](Graph &g) {
            auto *ctx = g.ctx();
            auto *hidden = subsample(g, g.dynamic_input(0));
            for (const auto &layer : layers) {
                for (int ffn = 1; ffn <= 2; ++ffn) {
                    const auto &branch = ffn == 1 ? layer.ffn1 : layer.ffn2;
                    auto value = g.linear(g.norm(hidden, branch.norm, 1e-5f), branch.input);
                    value = ggml_silu(ctx, value);
                    value = g.linear(value, branch.output);
                    hidden = ggml_add(ctx, hidden, ggml_scale(ctx, value, 0.5f));
                    if (ffn == 1) {
                        value = attention(g, g.norm(hidden, layer.attention_norm, 1e-5f), layer.attention);
                        hidden = ggml_add(ctx, hidden, value);
                        value = conformer_convolution(g, hidden, layer.convolution);
                        hidden = ggml_add(ctx, hidden, value);
                    }
                }
                hidden = g.norm(hidden, layer.final_norm, 1e-5f);
            }
            hidden = g.norm(hidden, final_norm, h.epsilon);
            return std::vector<ggml_tensor *>{hidden};
        },
        size_t(mel.rows) * 8);
    auto &output = outputs.front();
    return {std::move(output.values), int(output.rows), int(output.columns)};
}
} // namespace clap::detail
