#include "models.h"
#include "mert2-fft.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

namespace clap::detail {
using json = nlohmann::json;
namespace {

float sigmoid(float value) {
    if (value >= 0) return 1.0f / (1.0f + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}
} // namespace

Mert2Model::Mert2Model(const ModelLoader &source, int threads)
    : loader(source), ops(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
}

void Mert2Model::load_arch_hparams() {
    config = json::parse(loader.metadata("mert2.config"));
    if (config.at("model_type") != "mert2")
        throw std::runtime_error("CLAP: not a MERT2 model");
    if (config.at("variant") != "fs")
        throw std::runtime_error("CLAP: only MERT-v2-FullSong variant 'fs' is supported");
    const int hidden = config.at("hidden_size"), intermediate = config.at("intermediate_size");
    const int layers = config.at("num_hidden_layers"), heads = config.at("num_attention_heads");
    if (!(hidden > 0 && intermediate > 0 && layers > 0 && heads > 0 && hidden % heads == 0 &&
          (hidden / heads) % 2 == 0)) throw std::runtime_error("CLAP: invalid MERT2 encoder dimensions");
    const int bins = config.at("num_mel_bins"), rate = config.at("sampling_rate");
    const int fft_size = config.at("n_fft"), window = config.at("win_length"), hop = config.at("hop_length");
    if (!(bins > 0 && rate > 0 && fft_size > 0 && (fft_size & (fft_size - 1)) == 0 &&
          window == fft_size && hop > 0))
        throw std::runtime_error("CLAP: unsupported MERT2 waveform frontend geometry");
    if (config.at("minimum_input_samples").get<int>() != fft_size / 2 + 1)
        throw std::runtime_error("CLAP: invalid MERT2 minimum input length");
    if (config.at("inputs_to_logits_ratio").get<int>() != hop * 4)
        throw std::runtime_error("CLAP: invalid MERT2 frame ratio");
    const auto channels = config.at("subsampling_channels");
    const auto depths = config.at("subsampling_depths");
    if (!(channels.size() == 3 && depths.size() == 3 && channels[0] == bins && channels[2] == hidden))
        throw std::runtime_error("CLAP: invalid MERT2 subsampling layout");
    for (size_t index = 0; index < 3; ++index)
        if (!(channels[index].get<int>() > 0 && depths[index].get<int>() > 0))
            throw std::runtime_error("CLAP: invalid MERT2 subsampling dimensions");
    const int convolution = config.at("conv_depthwise_kernel_size");
    if (!(convolution > 0 && convolution % 2 == 1))
        throw std::runtime_error("CLAP: invalid MERT2 Conformer convolution kernel");
    for (const auto *name : {"layer_norm_eps", "subsampling_layer_norm_eps"}) {
        const float epsilon = config.at(name);
        if (!(std::isfinite(epsilon) && epsilon > 0))
            throw std::runtime_error(std::string("CLAP: invalid MERT2 ") + name);
    }
    if (!(config.at("rotary_embedding_base").get<float>() > 0))
        throw std::runtime_error("CLAP: invalid MERT2 rotary embedding base");
    const float context = config.at("context_seconds");
    if (!(std::isfinite(context) && context >= 30.0f))
        throw std::runtime_error("CLAP: MERT-v2-FullSong context metadata must cover at least 30 seconds");
}

void Mert2Model::load_arch_tensors() const {
    const int bins = config.at("num_mel_bins"), frequencies = config.at("n_fft").get<int>() / 2 + 1;
    loader.shape("feature_extractor.mel_mean", {bins});
    loader.shape("feature_extractor.mel_std", {bins});
    loader.shape("feature_extractor.spectrogram.window", {config.at("win_length").get<int>()});
    loader.shape("feature_extractor.mel_scale.fb", {bins, frequencies});

    const auto channels = config.at("subsampling_channels");
    const auto depths = config.at("subsampling_depths");
    int input = bins;
    const float ignored = config.at("subsampling_layer_norm_eps");
    (void) ignored;
    for (int stage = 0; stage < 3; ++stage) {
        const int output = channels[stage];
        const std::string root = "subsampling_module." + std::to_string(stage);
        if (stage > 0) {
            loader.norm_shape(root + ".resampling_layer.0", input);
            loader.shape(root + ".resampling_layer.2.weight", {2, input, output});
            loader.shape(root + ".resampling_layer.2.bias", {output});
        }
        for (int layer = 0; layer < depths[stage].get<int>(); ++layer) {
            const std::string prefix = root + ".convnext_layers." + std::to_string(layer);
            loader.shape(prefix + ".depthwise_block.1.weight", {7, 1, output});
            loader.shape(prefix + ".depthwise_block.1.bias", {output});
            loader.norm_shape(prefix + ".pointwise_block.0", output);
            loader.linear_shape(prefix + ".pointwise_block.1", output, output * 4);
            loader.shape(prefix + ".pointwise_block.3.weight", {output * 4, 1, 1});
            loader.shape(prefix + ".pointwise_block.3.bias", {output * 4, 1, 1});
            loader.linear_shape(prefix + ".pointwise_block.4", output * 4, output);
        }
        input = output;
    }

    const int hidden = config.at("hidden_size"), intermediate = config.at("intermediate_size");
    const int kernel = config.at("conv_depthwise_kernel_size");
    for (int layer = 0; layer < config.at("num_hidden_layers").get<int>(); ++layer) {
        const std::string prefix = "layers." + std::to_string(layer);
        for (const auto *name : {"ffn1_layer_norm", "attn_layer_norm", "ffn2_layer_norm", "final_layer_norm"})
            loader.norm_shape(prefix + "." + name, hidden);
        for (const auto *ffn : {"ffn1", "ffn2"}) {
            loader.linear_shape(prefix + "." + ffn + ".w_1", hidden, intermediate);
            loader.linear_shape(prefix + "." + ffn + ".w_2", intermediate, hidden);
        }
        for (const auto *projection : {"query_proj", "key_proj", "value_proj", "out_proj"})
            loader.linear_shape(prefix + ".attn." + projection, hidden, hidden);
        loader.norm_shape(prefix + ".conv_module.layer_norm", hidden);
        loader.shape(prefix + ".conv_module.conv_block.1.weight", {1, hidden, hidden * 2});
        loader.shape(prefix + ".conv_module.conv_block.3.weight", {kernel, 1, hidden});
        loader.norm_shape(prefix + ".conv_module.conv_block.4.1", hidden);
        loader.shape(prefix + ".conv_module.conv_block.6.weight", {1, hidden, hidden});
    }
}

TextParameters Mert2Model::text_parameters() const { return {0, 0, 0}; }
int Mert2Model::projection_dimension() const { return config.at("hidden_size"); }
int Mert2Model::mel_bins() const { return config.at("num_mel_bins"); }
int Mert2Model::sample_rate() const { return config.at("sampling_rate"); }
bool Mert2Model::is_waveform_encoder() const { return true; }

std::vector<float> Mert2Model::text(const std::vector<int> &) const {
    throw std::runtime_error("CLAP: MERT-v2-FullSong has no text encoder");
}


Matrix Mert2Model::frontend(const std::vector<float> &pcm) const {
    const int fft_size = config.at("n_fft"), hop = config.at("hop_length");
    const int bins = config.at("num_mel_bins"), frequencies = fft_size / 2 + 1;
    const int frames = static_cast<int>(pcm.size() / static_cast<size_t>(hop));
    if (!(frames > 0))
        throw std::runtime_error("CLAP: MERT2 waveform produces no feature frames");
    Matrix mel(frames, bins);
    const auto *window = loader.weight("feature_extractor.spectrogram.window");
    const auto *filters = loader.weight("feature_extractor.mel_scale.fb");
    const auto *mean = loader.weight("feature_extractor.mel_mean");
    const auto *standard_deviation = loader.weight("feature_extractor.mel_std");
    std::vector<std::complex<float>> spectrum(fft_size);
    std::vector<float> power(frequencies);
    for (int frame = 0; frame < frames; ++frame) {
        for (int index = 0; index < fft_size; ++index) {
            int source = frame * hop + index - fft_size / 2;
            if (source < 0) source = -source;
            if (source >= static_cast<int>(pcm.size())) source = 2 * static_cast<int>(pcm.size()) - 2 - source;
            spectrum[index] = pcm[source] * scalar(window, index);
        }
        mert2_fft(spectrum);
        for (int frequency = 0; frequency < frequencies; ++frequency)
            power[frequency] = std::norm(spectrum[frequency]);
        for (int bin = 0; bin < bins; ++bin) {
            float value = 0;
            for (int frequency = 0; frequency < frequencies; ++frequency)
                value += power[frequency] * scalar(filters, static_cast<size_t>(frequency) * bins + bin);
            value = 10.0f * std::log10(std::max(value, 1e-10f));
            mel[frame][bin] = (value - scalar(mean, bin)) /
                              std::max(scalar(standard_deviation, bin), 1e-5f);
        }
    }
    return mel;
}

Matrix Mert2Model::subsample(Matrix hidden) const {
    const auto channels = config.at("subsampling_channels");
    const auto depths = config.at("subsampling_depths");
    const float epsilon = config.at("subsampling_layer_norm_eps");
    for (int stage = 0; stage < 3; ++stage) {
        const int width = channels[stage];
        const std::string root = "subsampling_module." + std::to_string(stage);
        if (stage > 0) {
            hidden = ops.norm(hidden, root + ".resampling_layer.0", epsilon);
            hidden = ops.conv1d(hidden, root + ".resampling_layer.2", 2, 2, 0, 1);
        }
        for (int layer = 0; layer < depths[stage].get<int>(); ++layer) {
            const std::string prefix = root + ".convnext_layers." + std::to_string(layer);
            Matrix residual = hidden;
            auto value = ops.conv1d(hidden, prefix + ".depthwise_block.1", 7, 1, 3, width);
            value = ops.norm(value, prefix + ".pointwise_block.0", epsilon);
            value = ops.linear(value, prefix + ".pointwise_block.1");
            gelu(value);
            std::vector<double> magnitude(value.cols);
            for (int column = 0; column < value.cols; ++column) {
                for (int row = 0; row < value.rows; ++row)
                    magnitude[column] += static_cast<double>(value[row][column]) * value[row][column];
                magnitude[column] = std::sqrt(magnitude[column]);
            }
            double average = 0;
            for (double item : magnitude) average += item;
            average /= value.cols;
            const auto *weight = loader.weight(prefix + ".pointwise_block.3.weight");
            const auto *bias = loader.weight(prefix + ".pointwise_block.3.bias");
            for (int row = 0; row < value.rows; ++row) for (int column = 0; column < value.cols; ++column) {
                const float normalized = static_cast<float>(magnitude[column] / (average + 1e-6));
                value[row][column] = scalar(weight, column) * value[row][column] * normalized +
                                     scalar(bias, column) + value[row][column];
            }
            value = ops.linear(value, prefix + ".pointwise_block.4");
            add(value, residual);
            hidden = std::move(value);
        }
    }
    return hidden;
}

Matrix Mert2Model::attention(const Matrix &hidden, const std::string &prefix) const {
    Matrix query = ops.linear(hidden, prefix + ".query_proj");
    Matrix key = ops.linear(hidden, prefix + ".key_proj");
    const Matrix value = ops.linear(hidden, prefix + ".value_proj");
    const int heads = config.at("num_attention_heads"), head_dimension = hidden.cols / heads;
    const int half = head_dimension / 2;
    const float base = config.at("rotary_embedding_base");
    for (int row = 0; row < hidden.rows; ++row) for (int head = 0; head < heads; ++head) {
        const int offset = head * head_dimension;
        for (int column = 0; column < half; ++column) {
            const float inverse = std::pow(base, -2.0f * column / head_dimension);
            const float cosine = std::cos(row * inverse), sine = std::sin(row * inverse);
            const float q_first = query[row][offset + column], q_second = query[row][offset + half + column];
            const float k_first = key[row][offset + column], k_second = key[row][offset + half + column];
            query[row][offset + column] = q_first * cosine - q_second * sine;
            query[row][offset + half + column] = q_second * cosine + q_first * sine;
            key[row][offset + column] = k_first * cosine - k_second * sine;
            key[row][offset + half + column] = k_second * cosine + k_first * sine;
        }
    }
    Matrix attended(hidden.rows, hidden.cols);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
    std::vector<float> scores(hidden.rows);
    for (int head = 0; head < heads; ++head) for (int row = 0; row < hidden.rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int source = 0; source < hidden.rows; ++source) {
            float score = 0;
            for (int column = 0; column < head_dimension; ++column)
                score += query[row][head * head_dimension + column] * key[source][head * head_dimension + column];
            scores[source] = score * scale;
            maximum = std::max(maximum, scores[source]);
        }
        float denominator = 0;
        for (float &score : scores) { score = std::exp(score - maximum); denominator += score; }
        for (int source = 0; source < hidden.rows; ++source) {
            const float probability = scores[source] / denominator;
            for (int column = 0; column < head_dimension; ++column)
                attended[row][head * head_dimension + column] +=
                    probability * value[source][head * head_dimension + column];
        }
    }
    return ops.linear(attended, prefix + ".out_proj");
}

Matrix Mert2Model::conformer_convolution(const Matrix &hidden, const std::string &prefix) const {
    const int width = config.at("hidden_size"), kernel = config.at("conv_depthwise_kernel_size");
    auto value = ops.norm(hidden, prefix + ".layer_norm", config.at("layer_norm_eps"));
    value = ops.conv1d(value, prefix + ".conv_block.1", 1, 1, 0, 1, false);
    Matrix gated(value.rows, width);
    for (int row = 0; row < value.rows; ++row) for (int column = 0; column < width; ++column)
        gated[row][column] = value[row][column] * sigmoid(value[row][width + column]);
    value = ops.conv1d(gated, prefix + ".conv_block.3", kernel, 1, (kernel - 1) / 2, width, false);
    value = ops.norm(value, prefix + ".conv_block.4.1", config.at("layer_norm_eps"));
    gelu(value);
    return ops.conv1d(value, prefix + ".conv_block.6", 1, 1, 0, 1, false);
}

AudioFrames Mert2Model::audio_frames(const std::vector<float> &pcm) const {
    const size_t maximum = (size_t{1} << 30) / sizeof(float);
    if (!(pcm.size() >= config.at("minimum_input_samples").get<size_t>()))
        throw std::runtime_error("CLAP: MERT2 waveform is shorter than minimum_input_samples");
    if (!(pcm.size() <= maximum))
        throw std::runtime_error("CLAP: MERT2 waveform exceeds 1 GiB");
    if (!(std::all_of(pcm.begin(), pcm.end(), [](float value) { return std::isfinite(value); })))
        throw std::runtime_error("CLAP: MERT2 waveform contains a non-finite sample");
    Matrix hidden = subsample(frontend(pcm));
    const float epsilon = config.at("layer_norm_eps");
    for (int layer = 0; layer < config.at("num_hidden_layers").get<int>(); ++layer) {
        const std::string prefix = "layers." + std::to_string(layer);
        auto branch = ops.linear(ops.norm(hidden, prefix + ".ffn1_layer_norm", epsilon), prefix + ".ffn1.w_1");
        gelu(branch);
        branch = ops.linear(branch, prefix + ".ffn1.w_2");
        for (size_t index = 0; index < hidden.data.size(); ++index) hidden.data[index] += 0.5f * branch.data[index];
        branch = attention(ops.norm(hidden, prefix + ".attn_layer_norm", epsilon), prefix + ".attn");
        add(hidden, branch);
        branch = conformer_convolution(hidden, prefix + ".conv_module");
        add(hidden, branch);
        branch = ops.linear(ops.norm(hidden, prefix + ".ffn2_layer_norm", epsilon), prefix + ".ffn2.w_1");
        gelu(branch);
        branch = ops.linear(branch, prefix + ".ffn2.w_2");
        for (size_t index = 0; index < hidden.data.size(); ++index) hidden.data[index] += 0.5f * branch.data[index];
        hidden = ops.norm(hidden, prefix + ".final_layer_norm", epsilon);
    }
    return {std::move(hidden.data), hidden.rows, hidden.cols};
}
} // namespace clap::detail
