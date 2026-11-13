#include "audio.h"
#include "model-policy.h"
#include "models.h"
#include "quant.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <regex>
#include <stdexcept>

namespace clap::detail {
bool Mert2Model::tensor_quantizable(const ggml_tensor *w) {
    if (ggml_n_dims(w) != 2)
        return false;
    const std::string name(w->name);
    static const std::regex mert2(
        R"((subsampling_module\.[0-9]+\.convnext_layers\.[0-9]+\.pointwise_block\.(1|4)|layers\.[0-9]+\.(attn\.(query_proj|key_proj|value_proj|out_proj)|ffn[12]\.w_[12]))\.weight)");
    return std::regex_match(name, mert2);
}

TensorLoadPolicy Mert2Model::loading_policy(const ModelLoader &) {
    auto policy = encoder_loading_policy([](const ggml_tensor *w) {
        if (ggml_is_quantized(w->type))
            validate_encoder_storage(w, Mert2Model::tensor_quantizable(w));
    });
    return policy;
}

using json = nlohmann::json;
Mert2Model::Mert2Model(const ModelLoader &source, int threads) : loader(source), runtime(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
}

void Mert2Model::load_arch_hparams() {
    config = json::parse(loader.metadata("mert2.config"));
    if (config.at("model_type") != "mert2")
        throw std::runtime_error("CLAP: not a MERT2 model");
    if (config.at("variant") != "fs" && config.at("variant") != "30s")
        throw std::runtime_error("CLAP: MERT2 variant must be 'fs' or '30s'");
    const int hidden = config.at("hidden_size"), intermediate = config.at("intermediate_size");
    const int layers = config.at("num_hidden_layers"), heads = config.at("num_attention_heads");
    if (!(hidden > 0 && intermediate > 0 && layers > 0 && heads > 0 && hidden % heads == 0 &&
          (hidden / heads) % 2 == 0))
        throw std::runtime_error("CLAP: invalid MERT2 encoder dimensions");
    const int bins = config.at("num_mel_bins"), rate = config.at("sampling_rate");
    const int fft_size = config.at("n_fft"), window = config.at("win_length"), hop = config.at("hop_length");
    if (!(bins > 0 && rate > 0 && fft_size > 0 && (fft_size & (fft_size - 1)) == 0 && window == fft_size && hop > 0))
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
        throw std::runtime_error("CLAP: MERT2 context metadata must cover at least 30 seconds");
}

void Mert2Model::load_arch_tensors() {
    h = {config.at("hidden_size"),
         config.at("num_mel_bins"),
         config.at("sampling_rate"),
         config.at("n_fft"),
         config.at("hop_length"),
         config.at("num_attention_heads"),
         config.at("conv_depthwise_kernel_size"),
         config.at("minimum_input_samples"),
         config.at("layer_norm_eps"),
         config.at("subsampling_layer_norm_eps"),
         config.at("rotary_embedding_base")};
    const int bins = config.at("num_mel_bins"), frequencies = config.at("n_fft").get<int>() / 2 + 1;
    mean = loader.create_tensor("feature_extractor.mel_mean", {bins});
    stddev = loader.create_tensor("feature_extractor.mel_std", {bins});
    window = loader.create_tensor("feature_extractor.spectrogram.window", {config.at("win_length").get<int>()});
    filters = loader.create_tensor("feature_extractor.mel_scale.fb", {bins, frequencies});

    const auto channels = config.at("subsampling_channels");
    const auto depths = config.at("subsampling_depths");
    int input = bins;
    const float ignored = config.at("subsampling_layer_norm_eps");
    (void)ignored;
    for (int i = 0; i < 3; ++i) {
        Stage stage;
        const int output = channels[i];
        stage.width = output;
        const auto root = "subsampling_module." + std::to_string(i);
        if (i > 0) {
            stage.norm = create_norm(loader, root + ".resampling_layer.0", input);
            stage.resample = create_layer(loader, root + ".resampling_layer.2", {2, input, output}, {output});
        }
        for (int j = 0; j < depths[i].get<int>(); ++j) {
            const auto p = root + ".convnext_layers." + std::to_string(j);
            stage.blocks.push_back(
                {create_layer(loader, p + ".depthwise_block.1", {7, 1, output}, {output}),
                 create_norm(loader, p + ".pointwise_block.0", output),
                 create_linear(loader, p + ".pointwise_block.1", output, output * 4),
                 create_layer(loader, p + ".pointwise_block.3", {output * 4, 1, 1}, {output * 4, 1, 1}),
                 create_linear(loader, p + ".pointwise_block.4", output * 4, output)});
        }
        stages.push_back(std::move(stage));
        input = output;
    }

    const int hidden = config.at("hidden_size"), intermediate = config.at("intermediate_size");
    const int kernel = config.at("conv_depthwise_kernel_size");
    for (int i = 0; i < config.at("num_hidden_layers").get<int>(); ++i) {
        const auto p = "layers." + std::to_string(i);
        ConformerWeights layer;
        layer.ffn1.norm = create_norm(loader, p + ".ffn1_layer_norm", hidden);
        layer.attention_norm = create_norm(loader, p + ".attn_layer_norm", hidden);
        layer.ffn2.norm = create_norm(loader, p + ".ffn2_layer_norm", hidden);
        layer.final_norm = create_norm(loader, p + ".final_layer_norm", hidden);
        layer.ffn1.input = create_linear(loader, p + ".ffn1.w_1", hidden, intermediate);
        layer.ffn1.output = create_linear(loader, p + ".ffn1.w_2", intermediate, hidden);
        layer.ffn2.input = create_linear(loader, p + ".ffn2.w_1", hidden, intermediate);
        layer.ffn2.output = create_linear(loader, p + ".ffn2.w_2", intermediate, hidden);
        layer.attention = {create_linear(loader, p + ".attn.query_proj", hidden, hidden),
                           create_linear(loader, p + ".attn.key_proj", hidden, hidden),
                           create_linear(loader, p + ".attn.value_proj", hidden, hidden),
                           create_linear(loader, p + ".attn.out_proj", hidden, hidden)};
        layer.convolution.norm = create_norm(loader, p + ".conv_module.layer_norm", hidden);
        layer.convolution.input = create_layer(loader, p + ".conv_module.conv_block.1", {1, hidden, hidden * 2});
        layer.convolution.depthwise = create_layer(loader, p + ".conv_module.conv_block.3", {kernel, 1, hidden});
        layer.convolution.output_norm = create_norm(loader, p + ".conv_module.conv_block.4.1", hidden);
        layer.convolution.output = create_layer(loader, p + ".conv_module.conv_block.6", {1, hidden, hidden});
        layers.push_back(layer);
    }
}

TextParameters Mert2Model::text_parameters() const { return {0, 0, 0}; }
int Mert2Model::projection_dimension() const { return h.hidden; }
int Mert2Model::mel_bins() const { return h.bins; }
int Mert2Model::sample_rate() const { return h.sample_rate; }
bool Mert2Model::is_waveform_encoder() const { return true; }

std::vector<float> Mert2Model::text(const std::vector<int> &) const {
    throw std::runtime_error("CLAP: MERT2 has no text encoder");
}

HostMatrix Mert2Model::frontend(const std::vector<float> &pcm) const {
    const int fft_size = h.fft, hop = h.hop;
    const int bins = h.bins, frequencies = fft_size / 2 + 1;
    const int frames = static_cast<int>(pcm.size() / static_cast<size_t>(hop));
    if (!(frames > 0))
        throw std::runtime_error("CLAP: MERT2 waveform produces no feature frames");
    HostMatrix mel(frames, bins);
    std::vector<std::complex<double>> spectrum(fft_size);
    std::vector<float> power(frequencies);
    for (int frame = 0; frame < frames; ++frame) {
        for (int index = 0; index < fft_size; ++index) {
            int source = frame * hop + index - fft_size / 2;
            if (source < 0)
                source = -source;
            if (source >= static_cast<int>(pcm.size()))
                source = 2 * static_cast<int>(pcm.size()) - 2 - source;
            spectrum[index] = pcm[source] * scalar(window, index);
        }
        clap::fft(spectrum);
        for (int frequency = 0; frequency < frequencies; ++frequency)
            // Preserve MERT2's F32 FFT output rounding before power computation.
            power[frequency] = std::norm(static_cast<std::complex<float>>(spectrum[frequency]));
        for (int bin = 0; bin < bins; ++bin) {
            float value = 0;
            for (int frequency = 0; frequency < frequencies; ++frequency)
                value += power[frequency] * scalar(filters, static_cast<size_t>(frequency) * bins + bin);
            value = 10.0f * std::log10(std::max(value, 1e-10f));
            mel[frame][bin] = (value - scalar(mean, bin)) / std::max(scalar(stddev, bin), 1e-5f);
        }
    }
    return mel;
}

ggml_tensor *Mert2Model::subsample(Graph &g, ggml_tensor *hidden) const {
    auto *ctx = g.ctx();
    const float epsilon = h.subsampling_epsilon;
    for (int stage = 0; stage < 3; ++stage) {
        const auto &weights = stages[stage];
        const int width = weights.width;
        if (stage > 0) {
            hidden = g.norm(hidden, weights.norm, epsilon);
            hidden = g.conv1d(hidden, weights.resample, 2, 2, 0, 1);
        }
        for (const auto &block : weights.blocks) {
            auto *value = g.conv1d(hidden, block.depthwise, 7, 1, 3, width);
            value = g.norm(value, block.norm, epsilon);
            value = ggml_gelu_erf(ctx, g.linear(value, block.input));
            auto *time_major = ggml_cont(ctx, ggml_transpose(ctx, value));
            auto *magnitude =
                ggml_cont(ctx, ggml_transpose(ctx, ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, time_major)))));
            auto *average = ggml_add(ctx, ggml_mean(ctx, magnitude), g.input({1e-6f}, 1, 1));
            auto *normalized = ggml_div(ctx, magnitude, average);
            auto *scaled = ggml_mul(ctx, ggml_mul(ctx, value, g.weight(block.response.weight)), normalized);
            value = ggml_add(ctx, ggml_add(ctx, scaled, g.weight(block.response.bias)), value);
            value = g.linear(value, block.output);
            hidden = ggml_add(ctx, value, hidden);
        }
    }
    return hidden;
}

ggml_tensor *Mert2Model::attention(Graph &g, ggml_tensor *hidden, const AttentionWeights &weights) const {
    const int heads = h.heads;
    const float base = h.rotary_base;
    auto *query = g.rope(g.linear(hidden, weights.query), heads, base);
    auto *key = g.rope(g.linear(hidden, weights.key), heads, base);
    auto *value = g.linear(hidden, weights.value);
    return g.linear(g.attention(query, key, value, heads), weights.output);
}

ggml_tensor *Mert2Model::conformer_convolution(Graph &g, ggml_tensor *hidden, const ConvolutionWeights &weights) const {
    auto *ctx = g.ctx();
    const int width = h.hidden, kernel = h.kernel;
    auto *value = g.norm(hidden, weights.norm, h.epsilon);
    value = g.conv1d(value, weights.input, 1, 1, 0, 1);
    auto *gated = ggml_mul(ctx, g.slice(value, 0, width), ggml_sigmoid(ctx, g.slice(value, width, width)));
    value = g.conv1d(gated, weights.depthwise, kernel, 1, (kernel - 1) / 2, width);
    value = ggml_gelu_erf(ctx, g.norm(value, weights.output_norm, h.epsilon));
    return g.conv1d(value, weights.output, 1, 1, 0, 1);
}

AudioFrames Mert2Model::audio_frames(const std::vector<float> &pcm) const {
    const size_t maximum = (size_t{1} << 30) / sizeof(float);
    if (!(pcm.size() >= h.minimum_samples))
        throw std::runtime_error("CLAP: MERT2 waveform is shorter than minimum_input_samples");
    if (!(pcm.size() <= maximum))
        throw std::runtime_error("CLAP: MERT2 waveform exceeds 1 GiB");
    if (!(std::all_of(pcm.begin(), pcm.end(), [](float value) { return std::isfinite(value); })))
        throw std::runtime_error("CLAP: MERT2 waveform contains a non-finite sample");
    const auto mel = frontend(pcm);
    auto outputs = runtime.run_cached("frames", {GraphInput(mel)}, [&](Graph &g) {
        auto *ctx = g.ctx();
        auto *hidden = subsample(g, g.dynamic_input(0));
        const float epsilon = h.epsilon;
        for (const auto &layer : layers) {
            auto branch = g.linear(g.norm(hidden, layer.ffn1.norm, epsilon), layer.ffn1.input);
            branch = ggml_gelu_erf(ctx, branch);
            branch = g.linear(branch, layer.ffn1.output);
            hidden = ggml_add(ctx, hidden, ggml_scale(ctx, branch, 0.5f));
            branch = attention(g, g.norm(hidden, layer.attention_norm, epsilon), layer.attention);
            hidden = ggml_add(ctx, hidden, branch);
            branch = conformer_convolution(g, hidden, layer.convolution);
            hidden = ggml_add(ctx, hidden, branch);
            branch = g.linear(g.norm(hidden, layer.ffn2.norm, epsilon), layer.ffn2.input);
            branch = ggml_gelu_erf(ctx, branch);
            branch = g.linear(branch, layer.ffn2.output);
            hidden = ggml_add(ctx, hidden, ggml_scale(ctx, branch, 0.5f));
            hidden = g.norm(hidden, layer.final_norm, epsilon);
        }
        return std::vector<ggml_tensor *>{hidden};
    });
    auto &output = outputs.front();
    return {std::move(output.values), int(output.rows), int(output.columns)};
}
} // namespace clap::detail
