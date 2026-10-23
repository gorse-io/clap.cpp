#include "models.h"
#include "clap.h"
#include <ggml-cpu.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace clap::detail {
namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
float sigmoid(float value) {
    if (value >= 0) return 1.0f / (1.0f + std::exp(-value));
    const float e = std::exp(value); return e / (1.0f + e);
}
void silu(Matrix &x) { for (float &v : x.data) v *= sigmoid(v); }

}
MuqModel::MuqModel(const ModelLoader &source, int threads) : loader(source), ops(source, threads) {
    load_arch_hparams(); load_arch_tensors();
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
    const int64_t layer = 4*hidden*intermediate + 7*hidden*hidden + hidden*kernel + 64*hidden + 4*intermediate;
    const int64_t budget = 54*channels*channels + 32*channels*hidden + depth*layer;
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
    loader.shape("frontend.window", {2048}); loader.shape("frontend.filters", {128, 1025});
    const int channels = config.at("conv_dim"), hidden = config.at("encoder_dim");
    auto batch = [&](const std::string &name, int width) {
        loader.norm_shape(name, width);
        loader.shape(name + ".running_mean", {width}); loader.shape(name + ".running_var", {width});
    };
    for (int stage = 0; stage < 2; ++stage) {
        const std::string root = "conv.conv." + std::to_string(stage);
        for (int layer = 1; layer <= 3; ++layer) {
            const int input = stage == 0 && layer != 2 ? 1 : channels;
            const auto conv = root + ".conv" + std::to_string(layer);
            loader.shape(conv + ".weight", {3, 3, input, channels}); loader.shape(conv + ".bias", {channels});
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
        loader.shape(root + ".conv_module.depthwise_conv.weight", {encoder.at("conv_depthwise_kernel_size").get<int>(), 1, hidden});
        batch(root + ".conv_module.batch_norm", hidden);
        loader.shape(root + ".conv_module.pointwise_conv2.weight", {1, hidden, hidden});
    }
}
TextParameters MuqModel::text_parameters() const { return {0, 0, 0}; }
int MuqModel::projection_dimension() const { return config.at("encoder_dim"); }
int MuqModel::mel_bins() const { return 128; }
int MuqModel::sample_rate() const { return 24000; }
bool MuqModel::is_waveform_encoder() const { return true; }
std::vector<float> MuqModel::text(const std::vector<int> &) const {
    throw std::runtime_error("CLAP: MuQ has no text encoder");
}
Matrix MuqModel::frontend(const std::vector<float> &pcm) const {
    const int frames = static_cast<int>(pcm.size() / 240);
    Matrix mel(frames, 128);
    const auto *window = loader.weight("frontend.window"), *filters = loader.weight("frontend.filters");
    const float mean = config.at("stat").at("melspec_2048_mean"), stddev = config.at("stat").at("melspec_2048_std");
    std::vector<std::complex<double>> spectrum(2048);
    std::vector<float> power(1025);
    for (int frame = 0; frame < frames; ++frame) {
        for (int i = 0; i < 2048; ++i) {
            int source = frame * 240 + i - 1024;
            if (source < 0) source = -source;
            if (source >= static_cast<int>(pcm.size())) source = 2 * static_cast<int>(pcm.size()) - 2 - source;
            spectrum[i] = pcm[source] * scalar(window, i);
        }
        clap::fft(spectrum);
        for (int i = 0; i < 1025; ++i) power[i] = std::norm(spectrum[i]);
        for (int bin = 0; bin < 128; ++bin) {
            float value = 0;
            for (int frequency = 0; frequency < 1025; ++frequency)
                value += power[frequency] * scalar(filters, static_cast<size_t>(frequency) * 128 + bin);
            mel[frame][bin] = (10.0f * std::log10(std::max(value, 1e-10f)) - mean) / stddev;
        }
    }
    return mel;
}
void MuqModel::batch_norm(Matrix &x, const std::string &prefix) const {
    const auto *weight = loader.weight(prefix + ".weight"), *bias = loader.weight(prefix + ".bias");
    const auto *mean = loader.weight(prefix + ".running_mean"), *variance = loader.weight(prefix + ".running_var");
    for (int column = 0; column < x.cols; ++column) {
        const float scale = scalar(weight, column) / std::sqrt(scalar(variance, column) + 1e-5f);
        for (int row = 0; row < x.rows; ++row)
            x[row][column] = (x[row][column] - scalar(mean, column)) * scale + scalar(bias, column);
    }
}
MuqModel::Image MuqModel::conv2d(const Image &input, const std::string &prefix, int stride) const {
    const auto *weight = loader.weight(prefix + ".weight"), *bias = loader.weight(prefix + ".bias");
    const int frequencies = (input.frequencies + stride - 1) / stride;
    const int frames = (input.frames + stride - 1) / stride;
    const int columns = static_cast<int>(weight->ne[3]), reduction = input.values.cols * 9;
    Image result{frequencies, frames, Matrix(frequencies * frames, columns)};
    // Bound im2col scratch independently of song duration. Conv aliases are CPU
    // metadata only; do not cache their ephemeral addresses in the GPU backend.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> alias(ggml_init({4096, nullptr, true}), ggml_free);
    if (!(alias != nullptr))
        throw std::runtime_error("CLAP: MuQ convolution metadata allocation failed");
    auto *flat = ggml_new_tensor_2d(alias.get(), weight->type, reduction, columns);
    flat->data = weight->data;
    for (int start = 0; start < result.values.rows; start += 128) {
        const int rows = std::min(128, result.values.rows - start);
        const size_t bytes = (static_cast<size_t>(rows) * (reduction + columns)) * sizeof(float) + 2 * 1024 * 1024;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
        if (!(ctx != nullptr))
            throw std::runtime_error("CLAP: MuQ convolution allocation failed");
        auto *patches = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, reduction, rows);
        auto *data = static_cast<float *>(patches->data);
        std::fill_n(data, static_cast<size_t>(rows) * reduction, 0.0f);
        for (int row = 0; row < rows; ++row) {
            const int frequency = (start + row) / frames, time = (start + row) % frames;
            for (int channel = 0; channel < input.values.cols; ++channel)
                for (int y = 0; y < 3; ++y) for (int x = 0; x < 3; ++x) {
                    const int f = frequency * stride + y - 1, t = time * stride + x - 1;
                    if (f >= 0 && f < input.frequencies && t >= 0 && t < input.frames)
                        data[static_cast<size_t>(row) * reduction + channel * 9 + y * 3 + x] = input.values[f * input.frames + t][channel];
                }
        }
        auto *output = ggml_mul_mat(ctx.get(), flat, patches);
        auto *graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        auto plan = ggml_graph_plan(graph, ops.threads(), nullptr);
        std::vector<uint8_t> work(plan.work_size); plan.work_data = work.data();
        if (!(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS))
            throw std::runtime_error("CLAP: MuQ convolution compute failed");
        const auto *computed = static_cast<const float *>(output->data);
        for (int row = 0; row < rows; ++row) for (int column = 0; column < columns; ++column)
            result.values[start + row][column] = computed[row * columns + column] + scalar(bias, column);
    }
    return result;
}
Matrix MuqModel::subsample(const Matrix &mel) const {
    Image value{128, mel.rows, Matrix(128 * mel.rows, 1)};
    for (int f = 0; f < 128; ++f) for (int t = 0; t < mel.rows; ++t) value.values[f * mel.rows + t][0] = mel[t][f];
    for (int stage = 0; stage < 2; ++stage) {
        const std::string root = "conv.conv." + std::to_string(stage);
        auto residual = conv2d(value, root + ".conv3", 2); batch_norm(residual.values, root + ".bn3");
        value = conv2d(value, root + ".conv1", 2); batch_norm(value.values, root + ".bn1");
        for (float &v : value.values.data) v = std::max(v, 0.0f);
        value = conv2d(value, root + ".conv2", 1); batch_norm(value.values, root + ".bn2");
        add(value.values, residual.values);
        for (float &v : value.values.data) v = std::max(v, 0.0f);
    }
    Matrix flattened(value.frames, value.values.cols * value.frequencies);
    for (int t = 0; t < value.frames; ++t) for (int c = 0; c < value.values.cols; ++c)
        for (int f = 0; f < value.frequencies; ++f)
            flattened[t][c * value.frequencies + f] = value.values[f * value.frames + t][c];
    return ops.linear(flattened, "conv.linear");
}
Matrix MuqModel::attention(const Matrix &hidden, const std::string &prefix) const {
    const int heads = encoder.at("num_attention_heads"), dimension = hidden.cols / heads, half = dimension / 2;
    const float base = encoder.at("rotary_embedding_base");
    Matrix rotated = hidden;
    // Unlike MERT2, HF Wav2Vec2-Conformer rotates normalized hidden states
    // BEFORE both Q/K projections; V is projected from the unrotated states.
    for (int row = 0; row < hidden.rows; ++row) for (int head = 0; head < heads; ++head)
        for (int column = 0; column < half; ++column) {
            const float angle = row * std::pow(base, -2.0f * column / dimension);
            const float c = std::cos(angle), s = std::sin(angle);
            const int offset = head * dimension + column;
            rotated[row][offset] = hidden[row][offset] * c - hidden[row][offset + half] * s;
            rotated[row][offset + half] = hidden[row][offset + half] * c + hidden[row][offset] * s;
        }
    const auto query = ops.linear(rotated, prefix + ".linear_q"), key = ops.linear(rotated, prefix + ".linear_k");
    const auto value = ops.linear(hidden, prefix + ".linear_v");
    Matrix attended(hidden.rows, hidden.cols);
    std::vector<float> scores(hidden.rows);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    for (int head = 0; head < heads; ++head) for (int row = 0; row < hidden.rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int source = 0; source < hidden.rows; ++source) {
            float score = 0;
            for (int column = 0; column < dimension; ++column)
                score += query[row][head * dimension + column] * key[source][head * dimension + column];
            scores[source] = score * scale; maximum = std::max(maximum, scores[source]);
        }
        float denominator = 0;
        for (float &score : scores) { score = std::exp(score - maximum); denominator += score; }
        for (int source = 0; source < hidden.rows; ++source)
            for (int column = 0; column < dimension; ++column)
                attended[row][head * dimension + column] += scores[source] / denominator * value[source][head * dimension + column];
    }
    return ops.linear(attended, prefix + ".linear_out");
}
Matrix MuqModel::conformer_convolution(const Matrix &hidden, const std::string &prefix) const {
    const int width = hidden.cols, kernel = encoder.at("conv_depthwise_kernel_size");
    auto value = ops.norm(hidden, prefix + ".layer_norm", 1e-5f);
    value = ops.conv1d(value, prefix + ".pointwise_conv1", 1, 1, 0, 1, false);
    Matrix gated(value.rows, width);
    for (int row = 0; row < value.rows; ++row) for (int column = 0; column < width; ++column)
        gated[row][column] = value[row][column] * sigmoid(value[row][width + column]);
    value = ops.conv1d(gated, prefix + ".depthwise_conv", kernel, 1, (kernel - 1) / 2, width, false);
    batch_norm(value, prefix + ".batch_norm"); silu(value);
    return ops.conv1d(value, prefix + ".pointwise_conv2", 1, 1, 0, 1, false);
}
AudioFrames MuqModel::audio_frames(const std::vector<float> &pcm) const {
    if (!(pcm.size() >= 1025))
        throw std::runtime_error("CLAP: MuQ waveform must contain at least 1025 samples");
    if (!(pcm.size() <= (size_t{1} << 30) / sizeof(float)))
        throw std::runtime_error("CLAP: MuQ waveform exceeds 1 GiB");
    if (!(std::all_of(pcm.begin(), pcm.end(), [](float v) { return std::isfinite(v); })))
        throw std::runtime_error("CLAP: MuQ waveform contains non-finite samples");
    auto hidden = subsample(frontend(pcm));
    for (int layer = 0; layer < config.at("encoder_depth").get<int>(); ++layer) {
        const std::string root = "conformer.layers." + std::to_string(layer);
        for (int ffn = 1; ffn <= 2; ++ffn) {
            const std::string branch = root + ".ffn" + std::to_string(ffn);
            auto value = ops.linear(ops.norm(hidden, branch + "_layer_norm", 1e-5f), branch + ".intermediate_dense");
            silu(value); value = ops.linear(value, branch + ".output_dense");
            for (size_t i = 0; i < hidden.data.size(); ++i) hidden.data[i] += 0.5f * value.data[i];
            if (ffn == 1) {
                value = attention(ops.norm(hidden, root + ".self_attn_layer_norm", 1e-5f), root + ".self_attn");
                add(hidden, value);
                value = conformer_convolution(hidden, root + ".conv_module"); add(hidden, value);
            }
        }
        hidden = ops.norm(hidden, root + ".final_layer_norm", 1e-5f);
    }
    hidden = ops.norm(hidden, "conformer.layer_norm", encoder.at("layer_norm_eps"));
    return {std::move(hidden.data), hidden.rows, hidden.cols};
}
} // namespace clap::detail
