#include "audio.h"
#include "embeddinggemma2-audio.h"
#include "models.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

namespace clap::detail {
namespace {
// Names, RMS weights (already +1 in GGUF), NEOX RoPE and PLE order follow
// llama.cpp/src/models/gemma-embedding2.cpp, not the older Gemma embedding model.
constexpr const char *architecture = "gemma-embedding2.";
std::string key(const char *suffix) { return std::string(architecture) + suffix; }
[[noreturn]] void invalid(const std::string &field) {
    throw std::runtime_error("CLAP: invalid/unsupported EmbeddingGemma2 " + field);
}
int dimension(const nlohmann::json &value, const std::string &field, int maximum = 65536) {
    if (!value.is_number_integer() || value <= 0 || value > maximum)
        invalid(field);
    return value.get<int>();
}
void add(HostMatrix &x, const HostMatrix &y) {
    if (x.rows != y.rows || x.cols != y.cols)
        throw std::runtime_error("CLAP: Gemma residual shape mismatch");
    for (size_t i = 0; i < x.data.size(); ++i)
        x.data[i] += y.data[i];
}
void fast_gelu(HostMatrix &x) {
    // HF gelu_pytorch_tanh / GGML GELU, rather than erf GELU in older encoders.
    for (float &v : x.data)
        v = 0.5f * v * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
} // namespace

EmbeddingGemma2Model::EmbeddingGemma2Model(const ModelLoader &source, int threads, const ModelLoader *audio_loader)
    : loader(source), runtime(source, threads, true) {
    load_arch_hparams();
    load_arch_tensors();
    if (audio_loader) {
        if (audio_loader->integer("clip.audio.projection_dim") != hidden)
            invalid("audio projector dimension");
        const auto tokens = loader.metadata_value("tokenizer.ggml.tokens");
        for (size_t id = 0; id < tokens.size(); ++id) {
            if (tokens[id] == "<|audio>")
                begin_audio = static_cast<int>(id);
            if (tokens[id] == "<audio|>")
                end_audio = static_cast<int>(id);
        }
        if (begin_audio < 0 || end_audio < 0)
            invalid("audio delimiter tokens");
        audio_encoder = std::make_unique<GemmaAudioEncoder>(*audio_loader, threads);
    }
}
EmbeddingGemma2Model::~EmbeddingGemma2Model() = default;

HostMatrix EmbeddingGemma2Model::linear(const HostMatrix &x, const std::string &name, bool bias) const {
    Graph graph(runtime);
    auto *output = graph.linear(graph.input(x), name, bias);
    HostMatrix result(x.rows, static_cast<int>(output->ne[0]));
    result.data = graph.run(output);
    return result;
}

void EmbeddingGemma2Model::load_arch_hparams() {
    auto has = [&](const char *suffix) { return loader.has_metadata(key(suffix).c_str()); };
    auto value = [&](const char *suffix) { return loader.metadata_value(key(suffix).c_str()); };
    auto integer = [&](const char *suffix, int maximum = 65536) { return dimension(value(suffix), suffix, maximum); };
    auto optional = [&](const char *suffix, int fallback, int maximum = 65536) {
        return has(suffix) ? integer(suffix, maximum) : fallback;
    };
    auto number = [&](const char *suffix, double fallback) {
        if (!has(suffix))
            return static_cast<float>(fallback);
        const auto v = value(suffix);
        if (!v.is_number())
            invalid(suffix);
        const double n = v.get<double>();
        if (!std::isfinite(n) || n <= 0 || n > std::numeric_limits<float>::max() ||
            n < std::numeric_limits<float>::min())
            invalid(suffix);
        return static_cast<float>(n);
    };
    hidden = integer("embedding_length");
    layers = integer("block_count", 1024);
    output = optional("embedding_length_out", hidden);
    per_layer = integer("embedding_length_per_layer_input");
    context = integer("context_length", 1048576);
    window = integer("attention.sliding_window", 1048576);
    if (!has("attention.layer_norm_rms_epsilon"))
        invalid("attention.layer_norm_rms_epsilon");
    epsilon = number("attention.layer_norm_rms_epsilon", 1e-6);
    rope_base = number("rope.freq_base", 10000);
    rope_base_swa = number("rope.freq_base_swa", 10000);
    if (has("attention.causal") && (!value("attention.causal").is_boolean() || value("attention.causal").get<bool>()))
        invalid("causal attention");
    if (has("pooling_type") && (!value("pooling_type").is_number_integer() || value("pooling_type") != 1))
        invalid("pooling_type (mean required)");
    if (has("attention.shared_kv_layers") &&
        (!value("attention.shared_kv_layers").is_number_integer() || value("attention.shared_kv_layers") != 0))
        invalid("shared KV layers");
    if (has("rope.scaling.attn_factor") && number("rope.scaling.attn_factor", 1) != 1)
        invalid("rope.scaling.attn_factor");
    if (has("rope.scaling.type")) {
        const auto type = value("rope.scaling.type");
        if (type != "none" && type != "linear")
            invalid("rope.scaling.type");
        if (type == "linear") {
            if (!has("rope.scaling.factor"))
                invalid("rope.scaling.factor");
            rope_scale = 1.0f / number("rope.scaling.factor", 1);
        } else if (has("rope.scaling.factor") && number("rope.scaling.factor", 1) != 1) {
            invalid("rope.scaling.factor");
        }
    } else if (has("rope.scaling.factor")) {
        rope_scale = 1.0f / number("rope.scaling.factor", 1);
    }
    if (!std::isfinite(rope_scale) || rope_scale <= 0)
        invalid("rope frequency scale");
    const auto tokens = loader.metadata_value("tokenizer.ggml.tokens");
    if (!tokens.is_array() || tokens.empty() || tokens.size() > 1048576)
        invalid("vocabulary");
    vocab = static_cast<int>(tokens.size());
    auto special = [&](const char *name, int fallback) {
        if (!loader.has_metadata(name))
            return fallback;
        const auto v = loader.metadata_value(name);
        if (!v.is_number_integer() || v < 0 || v >= vocab)
            invalid(name);
        return v.get<int>();
    };
    bos = special("tokenizer.ggml.bos_token_id", 2);
    eos = special("tokenizer.ggml.eos_token_id", 1);
    if (bos >= vocab || eos >= vocab)
        invalid("special token ids");
    pad = special("tokenizer.ggml.padding_token_id", 0);
    (void)special("tokenizer.ggml.unknown_token_id", 3);
    auto per_block = [&](const char *suffix, int fallback, bool required) {
        std::vector<int> result(layers, fallback);
        if (!has(suffix)) {
            if (required)
                invalid(suffix);
            return result;
        }
        const auto v = value(suffix);
        if (v.is_array()) {
            if (v.size() != static_cast<size_t>(layers))
                invalid(suffix);
            for (int i = 0; i < layers; ++i)
                result[i] = dimension(v[i], suffix);
        } else {
            std::fill(result.begin(), result.end(), dimension(v, suffix));
        }
        return result;
    };
    heads = per_block("attention.head_count", 0, true);
    kv_heads = has("attention.head_count_kv") ? per_block("attention.head_count_kv", 0, true) : heads;
    ffn = per_block("feed_forward_length", 0, true);
    local.resize(layers);
    if (has("attention.sliding_window_pattern") && value("attention.sliding_window_pattern").is_array()) {
        const auto pattern = value("attention.sliding_window_pattern");
        if (pattern.size() != static_cast<size_t>(layers))
            invalid("sliding window pattern length");
        for (int i = 0; i < layers; ++i) {
            if (pattern[i].is_boolean())
                local[i] = pattern[i].get<bool>();
            else if (pattern[i].is_number_integer() && (pattern[i] == 0 || pattern[i] == 1))
                local[i] = pattern[i] == 1;
            else
                invalid("sliding window pattern element");
        }
    } else {
        const int pattern = optional("attention.sliding_window_pattern", 6, 1048576);
        for (int i = 0; i < layers; ++i)
            local[i] = (i + 1) % pattern != 0;
    }
    const int full_dimension = optional("attention.key_length", hidden / heads[0]);
    const int full_value = optional("attention.value_length", hidden / heads[0]);
    const int local_dimension = optional("attention.key_length_swa", full_dimension);
    const int local_value = optional("attention.value_length_swa", full_value);
    const int full_rotation = optional("rope.dimension_count", full_dimension);
    const int local_rotation = optional("rope.dimension_count_swa", full_rotation);
    if (full_dimension != full_value || local_dimension != local_value)
        invalid("key/value head dimension");
    dimensions.resize(layers);
    rotations.resize(layers);
    if (static_cast<int64_t>(per_layer) * layers > std::numeric_limits<int>::max() ||
        static_cast<int64_t>(hidden) * vocab > std::numeric_limits<int>::max())
        invalid("derived tensor products");
    for (int i = 0; i < layers; ++i) {
        dimensions[i] = local[i] ? local_dimension : full_dimension;
        rotations[i] = local[i] ? local_rotation : full_rotation;
        if (dimensions[i] <= 0 || heads[i] % kv_heads[i] || rotations[i] % 2 || rotations[i] > dimensions[i] ||
            static_cast<int64_t>(heads[i]) * dimensions[i] > 1048576 ||
            static_cast<int64_t>(hidden) * ffn[i] > std::numeric_limits<int>::max())
            invalid("attention/FFN geometry");
    }
}

void EmbeddingGemma2Model::load_arch_tensors() const {
    loader.shape("token_embd.weight", {hidden, vocab});
    loader.linear_shape("per_layer_model_proj", hidden, per_layer * layers, false);
    loader.shape("per_layer_proj_norm.weight", {per_layer});
    loader.shape("output_norm.weight", {hidden});
    loader.linear_shape("output", hidden, output, false);
    for (int i = 0; i < layers; ++i) {
        const auto root = "blk." + std::to_string(i) + ".";
        for (const auto *name : {"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "post_norm"})
            loader.shape(root + name + ".weight", {hidden});
        loader.shape(root + "attn_q_norm.weight", {dimensions[i]});
        loader.shape(root + "attn_k_norm.weight", {dimensions[i]});
        loader.linear_shape(root + "attn_q", hidden, heads[i] * dimensions[i], false);
        loader.linear_shape(root + "attn_k", hidden, kv_heads[i] * dimensions[i], false);
        loader.linear_shape(root + "attn_v", hidden, kv_heads[i] * dimensions[i], false);
        loader.linear_shape(root + "attn_output", heads[i] * dimensions[i], hidden, false);
        loader.linear_shape(root + "ffn_gate", hidden, ffn[i], false);
        loader.linear_shape(root + "ffn_up", hidden, ffn[i], false);
        loader.linear_shape(root + "ffn_down", ffn[i], hidden, false);
        loader.linear_shape(root + "inp_gate", hidden, per_layer, false);
        loader.linear_shape(root + "proj", per_layer, hidden, false);
        loader.shape(root + "layer_output_scale.weight", {1});
    }
}

TextParameters EmbeddingGemma2Model::text_parameters() const { return {bos, eos, static_cast<size_t>(context)}; }
AudioFrames EmbeddingGemma2Model::audio_frames(const std::vector<float> &) const {
    throw std::runtime_error("CLAP: EmbeddingGemma2 does not expose waveform frames; use encode_audio");
}

std::vector<float> EmbeddingGemma2Model::waveform_embedding(const std::vector<float> &pcm, size_t crop_offset) const {
    if (!audio_encoder)
        throw std::runtime_error("CLAP: EmbeddingGemma2 audio requires an mmproj GGUF");
    if (crop_offset)
        invalid("audio crop offset (unsupported)");
    const HostMatrix soft = audio_encoder->encode(pcm);
    if (soft.cols != hidden || soft.rows > context - 4)
        invalid("audio soft-token dimensions");
    HostMatrix merged(soft.rows + 4, hidden);
    const auto *embedding = loader.weight("token_embd.weight");
    const float scale = std::sqrt(static_cast<float>(hidden));
    const int ids[] = {bos, begin_audio, end_audio, eos};
    const int positions[] = {0, 1, soft.rows + 2, soft.rows + 3};
    for (int i = 0; i < 4; ++i) {
        tensor_row(embedding, ids[i], merged[positions[i]]);
        for (int c = 0; c < hidden; ++c)
            merged[positions[i]][c] *= scale;
    }
    // Projected audio tokens already inhabit the scaled embedding space.
    // Unlike hard tokens, they must not be multiplied by sqrt(hidden).
    for (int row = 0; row < soft.rows; ++row)
        std::copy_n(soft[row], hidden, merged[row + 2]);
    std::vector<int> mask(merged.rows, 1);
    return encode(std::move(merged), mask);
}

HostMatrix EmbeddingGemma2Model::rms(const HostMatrix &x, const std::string &name, int width) const {
    if (!width)
        width = x.cols;
    HostMatrix out(x.rows, x.cols);
    const auto *weight = name.empty() ? nullptr : loader.weight(name + ".weight");
    // GGUF norm weights already contain HF's +1; never add it again.
    std::vector<float> scales(width, 1.0f);
    if (weight)
        tensor_row(weight, 0, scales.data());
    for (int row = 0; row < x.rows; ++row)
        for (int offset = 0; offset < x.cols; offset += width) {
            float square = 0;
            for (int c = 0; c < width; ++c)
                square += x[row][offset + c] * x[row][offset + c];
            const float scale = 1.0f / std::sqrt(square / width + epsilon);
            for (int c = 0; c < width; ++c)
                out[row][offset + c] = x[row][offset + c] * scale * scales[c];
        }
    return out;
}

HostMatrix EmbeddingGemma2Model::attention(const HostMatrix &x, int layer, const std::vector<int> &mask) const {
    const auto root = "blk." + std::to_string(layer) + ".";
    const int dim = dimensions[layer], rot = rotations[layer];
    const auto normalized = rms(x, root + "attn_norm");
    HostMatrix q = rms(linear(normalized, root + "attn_q", false), root + "attn_q_norm", dim);
    HostMatrix k = rms(linear(normalized, root + "attn_k", false), root + "attn_k_norm", dim);
    const HostMatrix v = rms(linear(normalized, root + "attn_v", false), "", dim);
    const float base = local[layer] ? rope_base_swa : rope_base;
    const float frequency_scale = local[layer] ? 1.0f : rope_scale;
    // GGUF NEOX convention rotates the first n_rot channels; partial-RoPE
    // channels after n_rot remain untouched. Q/K checkpoint permutation is
    // performed by the GGUF converter, not here.
    for (HostMatrix *m : {&q, &k}) {
        const int count = m->cols / dim;
        for (int c = 0; c < rot / 2; ++c) {
            const float frequency = std::pow(base, -2.0f * c / rot) * frequency_scale;
            for (int row = 0; row < m->rows; ++row) {
                const float cosine = std::cos(row * frequency), sine = std::sin(row * frequency);
                for (int head = 0; head < count; ++head) {
                    float *p = (*m)[row] + head * dim;
                    const float a = p[c], b = p[c + rot / 2];
                    p[c] = a * cosine - b * sine;
                    p[c + rot / 2] = a * sine + b * cosine;
                }
            }
        }
    }
    HostMatrix result(x.rows, q.cols);
    // One score row at a time avoids a context^2 attention allocation.
    std::vector<float> scores(x.rows);
    for (int row = 0; row < x.rows; ++row) {
        // GGUF stores twice the HF radius. Symmetric SWA includes both
        // endpoints: abs(query_position - key_position) <= window / 2.
        const int begin = local[layer] ? std::max(0, row - window / 2) : 0;
        const int end = local[layer]
                            ? static_cast<int>(std::min<int64_t>(x.rows, static_cast<int64_t>(row) + window / 2 + 1))
                            : x.rows;
        for (int head = 0; head < heads[layer]; ++head) {
            const int kv_head = head / (heads[layer] / kv_heads[layer]);
            float maximum = -std::numeric_limits<float>::infinity();
            for (int pos = begin; pos < end; ++pos) {
                if (!mask[pos])
                    continue;
                float dot = 0;
                for (int c = 0; c < dim; ++c)
                    dot += q[row][head * dim + c] * k[pos][kv_head * dim + c];
                // Unit attention scale: Q and K are already learned RMS-normalized.
                scores[pos] = dot;
                maximum = std::max(maximum, dot);
            }
            float total = 0;
            for (int pos = begin; pos < end; ++pos)
                if (mask[pos])
                    total += (scores[pos] = std::exp(scores[pos] - maximum));
            for (int pos = begin; pos < end; ++pos) {
                if (!mask[pos])
                    continue;
                const float probability = scores[pos] / total;
                for (int c = 0; c < dim; ++c)
                    result[row][head * dim + c] += probability * v[pos][kv_head * dim + c];
            }
        }
    }
    return rms(linear(result, root + "attn_output", false), root + "post_attention_norm");
}

std::vector<float> EmbeddingGemma2Model::text(const std::vector<int> &ids) const {
    if (ids.empty() || ids.size() > static_cast<size_t>(context))
        invalid("token sequence length");
    for (int id : ids)
        if (id < 0 || id >= vocab)
            invalid("token id");
    std::vector<int> mask(ids.size());
    int real_tokens = 0;
    for (size_t i = 0; i < ids.size(); ++i)
        real_tokens += (mask[i] = ids[i] != pad);
    if (!real_tokens)
        invalid("all-padding token input");
    int64_t width = std::max<int64_t>(std::max(hidden, output), static_cast<int64_t>(per_layer) * layers);
    for (int i = 0; i < layers; ++i)
        width = std::max<int64_t>(width, std::max(heads[i] * dimensions[i], ffn[i]));
    if (ids.size() > static_cast<size_t>(std::numeric_limits<int>::max() / width))
        invalid("token matrix size");
    HostMatrix x(static_cast<int>(ids.size()), hidden);
    const auto *embedding = loader.weight("token_embd.weight");
    const float scale = std::sqrt(static_cast<float>(hidden));
    for (int row = 0; row < x.rows; ++row) {
        tensor_row(embedding, ids[row], x[row]);
        for (int c = 0; c < hidden; ++c)
            x[row][c] *= scale;
    }
    return encode(std::move(x), mask);
}

std::vector<float> EmbeddingGemma2Model::encode(HostMatrix x, const std::vector<int> &mask) const {
    if (x.cols != hidden || x.rows > context || mask.size() != static_cast<size_t>(x.rows))
        invalid("soft-token matrix shape");
    int real_tokens = 0;
    for (int valid : mask)
        real_tokens += valid != 0;
    if (!real_tokens)
        invalid("all-padding soft-token input");
    const float scale = std::sqrt(static_cast<float>(hidden));
    HostMatrix projected = linear(x, "per_layer_model_proj", false);
    for (float &v : projected.data)
        v /= scale;
    projected = rms(projected, "per_layer_proj_norm", per_layer);
    for (int layer = 0; layer < layers; ++layer) {
        const auto root = "blk." + std::to_string(layer) + ".";
        add(x, attention(x, layer, mask));
        const auto pre = rms(x, root + "ffn_norm");
        HostMatrix gate = linear(pre, root + "ffn_gate", false);
        fast_gelu(gate);
        const HostMatrix up = linear(pre, root + "ffn_up", false);
        for (size_t i = 0; i < gate.data.size(); ++i)
            gate.data[i] *= up.data[i];
        add(x, rms(linear(gate, root + "ffn_down", false), root + "post_ffw_norm"));
        HostMatrix ple = linear(x, root + "inp_gate", false);
        fast_gelu(ple);
        for (int row = 0; row < x.rows; ++row)
            for (int c = 0; c < per_layer; ++c)
                ple[row][c] *= projected[row][layer * per_layer + c];
        add(x, rms(linear(ple, root + "proj", false), root + "post_norm"));
        const float out_scale = scalar(loader.weight(root + "layer_output_scale.weight"), 0);
        for (float &v : x.data)
            v *= out_scale;
    }
    // Match the reference graph's operation order even for storage types that
    // pack/round the activation operand: projecting a pooled row first is only
    // algebraically equivalent, not equivalent under F16/Q8 input packing.
    const HostMatrix final = linear(rms(x, "output_norm"), "output", false);
    std::vector<float> result(output, 0.0f);
    for (int row = 0; row < final.rows; ++row)
        if (mask[row])
            for (int c = 0; c < output; ++c)
                result[c] += final[row][c] / real_tokens;
    double square = 0;
    for (float v : result) {
        if (!std::isfinite(v))
            invalid("non-finite output embedding");
        square += static_cast<double>(v) * v;
    }
    if (!(square > 0))
        invalid("zero output embedding");
    const float inverse = static_cast<float>(1.0 / std::sqrt(square));
    for (float &v : result)
        v *= inverse;
    return result;
}
} // namespace clap::detail

// Gemma4-audio Conformer contract: Hugging Face modeling_gemma4.py and
// llama.cpp tools/mtmd/models/gemma4a.cpp. Audio files/rate conversion belong
// exclusively to src/audio.cpp; this module accepts mono 16 kHz PCM.

namespace clap::detail {
namespace {
constexpr double pi = 3.14159265358979323846;
constexpr int mel_bins = 128, max_samples = 480000, left_context = 12;
void require(bool valid, const std::string &message) {
    if (!valid)
        throw std::runtime_error("CLAP: Gemma audio " + message);
}
void silu(HostMatrix &x) {
    for (auto &v : x.data)
        v = v / (1.0f + std::exp(-v));
}
void clip(HostMatrix &x, float lo, float hi) {
    for (auto &v : x.data)
        v = std::clamp(v, lo, hi);
}
void finite(const HostMatrix &x) {
    for (float v : x.data)
        require(std::isfinite(v), "non-finite activation");
}
// F32 im2col operand and decoded small convolution kernel. Unlike ggml_conv_2d,
// this does not implicitly pack the activation into F16 before convolution.
HostMatrix convolve(const HostMatrix &patches, const ggml_tensor *weight, int outputs, int threads) {
    HostMatrix result(patches.rows, outputs);
    const size_t count = static_cast<size_t>(patches.cols) * outputs;
    const size_t bytes = (patches.data.size() + result.data.size() + count) * sizeof(float) + 2 * 1024 * 1024;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
    require(bool(ctx), "convolution allocation failed");
    auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, patches.cols, patches.rows);
    auto *kernel = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, patches.cols, outputs);
    std::memcpy(input->data, patches.data.data(), patches.data.size() * sizeof(float));
    auto *decoded = static_cast<float *>(kernel->data);
    for (size_t i = 0; i < count; ++i)
        decoded[i] = scalar(weight, i);
    auto *output = ggml_mul_mat(ctx.get(), kernel, input);
    auto *graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    auto plan = ggml_graph_plan(graph, threads, nullptr);
    std::vector<uint8_t> workspace(plan.work_size);
    plan.work_data = workspace.data();
    require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "convolution compute failed");
    std::memcpy(result.data.data(), output->data, result.data.size() * sizeof(float));
    return result;
}
} // namespace

GemmaAudioEncoder::GemmaAudioEncoder(const ModelLoader &source, int threads)
    : loader(source), hidden(0), heads(0), layers(0), feedforward(0), channels{0, 0}, output(0), epsilon(0),
      runtime(source, threads, true),
      threads(threads > 0 ? threads
                          : static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())))) {
    require(loader.metadata("general.architecture") == "clip", "requires clip mmproj architecture");
    require(loader.boolean("clip.has_audio_encoder"), "encoder missing");
    require(loader.metadata("clip.audio.projector_type") == "gemma4a", "unsupported projector variant");
    auto dimension = [&](const char *key, int max) {
        const auto n = loader.integer(key);
        require(n > 0 && n <= max, std::string("invalid dimension: ") + key);
        return static_cast<int>(n);
    };
    hidden = dimension("clip.audio.embedding_length", 4096);
    heads = dimension("clip.audio.attention.head_count", 128);
    layers = dimension("clip.audio.block_count", 128);
    feedforward = dimension("clip.audio.feed_forward_length", 16384);
    require(hidden % heads == 0 && hidden % 2 == 0, "incompatible head dimensions");
    require(feedforward == 4 * hidden, "incompatible feed-forward dimension");
    require(loader.integer("clip.audio.num_mel_bins") == mel_bins, "requires 128 mel bins");
    const int projection = dimension("clip.audio.projection_dim", 16384);
    epsilon = static_cast<float>(loader.number("clip.audio.attention.layer_norm_epsilon"));
    require(std::isfinite(epsilon) && epsilon > 0, "invalid norm epsilon");
    for (int i = 0; i < 2; ++i) {
        const auto name = "a.conv1d." + std::to_string(i);
        const auto *w = loader.weight(name + ".weight");
        require(w->ne[3] > 0 && w->ne[3] <= 4096, "invalid convolution channels");
        channels[i] = static_cast<int>(w->ne[3]);
        loader.shape(name + ".weight", {3, 3, i ? channels[0] : 1, channels[i]});
        loader.shape(name + ".norm.weight", {channels[i]});
        require(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16,
                "subsampling kernels must be unquantized");
    }
    const auto *out = loader.weight("a.pre_encode.out.weight");
    require(out->ne[1] > 0 && out->ne[1] <= 8192, "invalid output dimension");
    output = static_cast<int>(out->ne[1]);
    loader.linear_shape("a.input_projection", 32 * channels[1], hidden, false);
    loader.linear_shape("a.pre_encode.out", hidden, output, true);
    loader.linear_shape("mm.a.input_projection", output, projection, false);
    if (loader.has_tensor_prefix("mm.a.soft_emb_norm.weight"))
        loader.shape("mm.a.soft_emb_norm.weight", {output});
    const auto norm = [&](const std::string &name, int n) {
        loader.shape(name + ".weight", {n});
        auto *w = loader.weight(name + ".weight");
        require(!ggml_is_quantized(w->type), "norm vectors must be unquantized: " + name);
        for (int i = 0; i < n; ++i)
            require(std::isfinite(scalar(w, i)), "non-finite norm: " + name);
    };
    for (int il = 0; il < layers; ++il) {
        const auto p = "a.blk." + std::to_string(il) + ".";
        for (const auto *name : {"ffn_norm", "ffn_post_norm", "ffn_norm_1", "ffn_post_norm_1", "norm_conv", "conv_norm",
                                 "ln2", "attn_pre_norm", "attn_post_norm"})
            norm(p + name, hidden);
        loader.shape(p + "conv_dw.weight", {5, hidden});
        require(!ggml_is_quantized(loader.weight(p + "conv_dw.weight")->type), "depthwise kernel must be unquantized");
        loader.shape(p + "per_dim_scale.weight", {hidden / heads});
        for (int d = 0; d < hidden / heads; ++d) {
            const float value = scalar(loader.weight(p + "per_dim_scale.weight"), d);
            require(std::isfinite(value) && value >= 0, "invalid converted per-dimension scale");
        }
        loader.linear_shape(p + "attn_k_rel", hidden, hidden, false);
        for (const auto *name : {"ffn_up", "ffn_down", "ffn_up_1", "ffn_down_1", "attn_q", "attn_k", "attn_v",
                                 "attn_out", "conv_pw1", "conv_pw2"}) {
            const std::string n = name;
            const bool up = n == "ffn_up" || n == "ffn_up_1";
            const bool down = n == "ffn_down" || n == "ffn_down_1";
            loader.linear_shape(p + n, down ? feedforward : hidden,
                                up                ? feedforward
                                : n == "conv_pw1" ? 2 * hidden
                                                  : hidden,
                                false);
            for (const auto *suffix : {"input_min", "input_max", "output_min", "output_max"}) {
                loader.shape(p + n + "." + suffix, {1});
                require(std::isfinite(scalar(loader.weight(p + n + "." + suffix), 0)), "non-finite clamp: " + p + n);
            }
            for (const auto *kind : {"input", "output"}) {
                const auto base = p + n + "." + kind;
                require(scalar(loader.weight(base + "_min"), 0) <= scalar(loader.weight(base + "_max"), 0),
                        "inverted clamp: " + p + n);
            }
        }
    }
    // The published file also carries vision weights, which are out of scope.
    // Reject unknown audio tensors without requiring coverage of that tower.
    loader.validate_tensor_coverage("a.");
    loader.validate_tensor_coverage("mm.a.");
}

HostMatrix GemmaAudioEncoder::linear(const HostMatrix &x, const std::string &name, bool bias) const {
    Graph graph(runtime);
    auto *output = graph.linear(graph.input(x), name, bias);
    HostMatrix result(x.rows, static_cast<int>(output->ne[0]));
    result.data = graph.run(output);
    return result;
}

HostMatrix GemmaAudioEncoder::log_mel(const std::vector<float> &pcm) {
    require(pcm.size() > 160, "input is too short (at least 161 samples required)");
    for (float v : pcm)
        require(std::isfinite(v), "non-finite PCM input");
    const int samples = static_cast<int>(std::min(pcm.size(), static_cast<size_t>(max_samples)));
    const int frames = (samples - 161) / 160 + 1;
    HostMatrix result(frames, mel_bins);
    std::array<double, 130> edges;
    for (int i = 0; i < 130; ++i)
        edges[i] = 700.0 * (std::pow(10.0, std::log10(1.0 + 8000.0 / 700.0) * i / 129.0) - 1.0);
    std::array<float, 320> window;
    for (int n = 0; n < 320; ++n)
        window[n] = static_cast<float>(0.5 - 0.5 * std::cos(2 * pi * n / 320));
    std::array<std::array<double, 257>, mel_bins> filters{};
    for (int m = 0; m < mel_bins; ++m)
        for (int k = 0; k < 257; ++k) {
            const double hz = k * 16000.0 / 512;
            filters[m][k] = std::max(0.0, std::min((hz - edges[m]) / (edges[m + 1] - edges[m]),
                                                   (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1])));
        }
    std::vector<std::complex<double>> spectrum(512);
    for (int t = 0; t < frames; ++t) {
        std::fill(spectrum.begin(), spectrum.end(), std::complex<double>{});
        for (int n = 0; n < 320; ++n) {
            const int i = t * 160 + n - 160;
            const float sample = i < 0 || i >= samples ? 0.0f : pcm[i] * window[n];
            spectrum[n] = sample;
        }
        clap::fft(spectrum);
        std::array<double, 257> magnitude;
        for (int k = 0; k < 257; ++k)
            magnitude[k] = std::abs(spectrum[k]);
        for (int m = 0; m < mel_bins; ++m) {
            double mel = 0;
            for (int k = 0; k < 257; ++k)
                mel += magnitude[k] * filters[m][k];
            result[t][m] = static_cast<float>(std::log(mel + 0.001));
        }
    }
    return result;
}

HostMatrix GemmaAudioEncoder::rms(const HostMatrix &x, const std::string &name) const {
    HostMatrix y(x.rows, x.cols);
    const auto *weight = name.empty() ? nullptr : loader.weight(name + ".weight");
    for (int r = 0; r < x.rows; ++r) {
        double sum = 0;
        for (int d = 0; d < x.cols; ++d)
            sum += double(x[r][d]) * x[r][d];
        const float scale = static_cast<float>(1.0 / std::sqrt(sum / x.cols + epsilon));
        for (int d = 0; d < x.cols; ++d)
            y[r][d] = x[r][d] * scale * (weight ? scalar(weight, d) : 1.0f);
    }
    return y;
}
HostMatrix GemmaAudioEncoder::clipped_linear(const HostMatrix &x, const std::string &name) const {
    HostMatrix input = x;
    clip(input, scalar(loader.weight(name + ".input_min"), 0), scalar(loader.weight(name + ".input_max"), 0));
    auto y = linear(input, name, false);
    clip(y, scalar(loader.weight(name + ".output_min"), 0), scalar(loader.weight(name + ".output_max"), 0));
    return y;
}
HostMatrix GemmaAudioEncoder::feed_forward(const HostMatrix &x, const std::string &p, const std::string &suffix) const {
    HostMatrix input = x;
    clip(input, -1e10f, 1e10f);
    auto y = clipped_linear(rms(input, p + "ffn_norm" + suffix), p + "ffn_up" + suffix);
    silu(y);
    y = clipped_linear(y, p + "ffn_down" + suffix);
    clip(y, -1e10f, 1e10f);
    y = rms(y, p + "ffn_post_norm" + suffix);
    for (size_t i = 0; i < y.data.size(); ++i)
        y.data[i] = x.data[i] + 0.5f * y.data[i];
    return y;
}
HostMatrix GemmaAudioEncoder::subsample(const HostMatrix &features) const {
    // Time-major [T,F,C]. Both convolutions use symmetric pad=1,stride=2.
    HostMatrix cur = features;
    int time = features.rows, freq = mel_bins, inc = 1;
    for (int layer = 0; layer < 2; ++layer) {
        const int nt = (time + 1) / 2, nf = (freq + 1) / 2, oc = channels[layer];
        HostMatrix patches(nt * nf, 9 * inc);
        for (int t = 0; t < nt; ++t)
            for (int f = 0; f < nf; ++f)
                for (int c = 0; c < inc; ++c)
                    for (int kt = 0; kt < 3; ++kt)
                        for (int kf = 0; kf < 3; ++kf) {
                            const int it = 2 * t + kt - 1, jf = 2 * f + kf - 1;
                            if (it >= 0 && it < time && jf >= 0 && jf < freq)
                                patches[t * nf + f][(c * 3 + kt) * 3 + kf] = cur[it][jf * inc + c];
                        }
        const auto p = "a.conv1d." + std::to_string(layer);
        auto convolved = convolve(patches, loader.weight(p + ".weight"), oc, threads);
        const auto *w = loader.weight(p + ".norm.weight");
        HostMatrix next(nt, nf * oc);
        for (int r = 0; r < nt * nf; ++r) {
            double mean = 0, var = 0;
            for (int c = 0; c < oc; ++c)
                mean += convolved[r][c];
            mean /= oc;
            for (int c = 0; c < oc; ++c) {
                const double diff = convolved[r][c] - mean;
                var += diff * diff;
            }
            const float scale = static_cast<float>(1.0 / std::sqrt(var / oc + epsilon));
            for (int c = 0; c < oc; ++c)
                next[r / nf][(r % nf) * oc + c] =
                    std::max(0.0f, (convolved[r][c] - static_cast<float>(mean)) * scale * scalar(w, c));
        }
        cur = std::move(next);
        time = nt;
        freq = nf;
        inc = oc;
    }
    return linear(cur, "a.input_projection", false);
}
HostMatrix GemmaAudioEncoder::attention(const HostMatrix &x, const std::string &p, const HostMatrix &positions) const {
    const int dim = hidden / heads;
    auto q = clipped_linear(x, p + "attn_q"), k = clipped_linear(x, p + "attn_k"), v = clipped_linear(x, p + "attn_v");
    const auto relative = linear(positions, p + "attn_k_rel", false);
    const float qs = static_cast<float>(1.0 / std::sqrt(double(dim)) / std::log(2.0));
    const float ks = static_cast<float>(std::log(1.0 + std::exp(1.0)) / std::log(2.0));
    const auto *scale = loader.weight(p + "per_dim_scale.weight");
    for (int t = 0; t < x.rows; ++t)
        for (int d = 0; d < hidden; ++d) {
            q[t][d] *= qs * scalar(scale, d % dim);
            k[t][d] *= ks;
        }
    HostMatrix y(x.rows, hidden);
    // Exact reduction of blocked [12,24] attention for right_context=0:
    // valid keys j are t-12<j<=t (the HF mask's left bound is strict).
    // Under the upstream relative shift,
    // (qi*24+s) / 25 == qi and (qi*24+s) % 25 == 12-(t-j).
    // Padded/invalid logits are -1e9 and have exactly zero softmax mass.
    for (int t = 0; t < x.rows; ++t)
        for (int h = 0; h < heads; ++h) {
            const int first = std::max(0, t - left_context + 1), count = t - first + 1;
            std::array<float, 13> scores;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int j = first; j <= t; ++j) {
                float content = 0, rpe = 0;
                for (int d = 0; d < dim; ++d) {
                    const int c = h * dim + d;
                    content += q[t][c] * k[j][c];
                    rpe += q[t][c] * relative[left_context - (t - j)][c];
                }
                scores[j - first] = 50.0f * std::tanh((content + rpe) / 50.0f);
                maximum = std::max(maximum, scores[j - first]);
            }
            float sum = 0;
            for (int j = 0; j < count; ++j) {
                scores[j] = std::exp(scores[j] - maximum);
                sum += scores[j];
            }
            for (int j = first; j <= t; ++j)
                for (int d = 0; d < dim; ++d)
                    y[t][h * dim + d] += scores[j - first] / sum * v[j][h * dim + d];
        }
    return clipped_linear(y, p + "attn_out");
}
HostMatrix GemmaAudioEncoder::encode_features(const HostMatrix &features) const {
    require(features.cols == mel_bins && features.rows > 0 && features.rows <= 3000, "invalid feature shape");
    finite(features);
    auto x = subsample(features);
    HostMatrix positions(13, hidden);
    const int half = hidden / 2;
    for (int r = 0; r < 13; ++r)
        for (int i = 0; i < half; ++i) {
            const float timescale = std::exp(static_cast<float>(-std::log(10000.0) * i / std::max(half - 1, 1)));
            const float phase = (12 - r) * timescale;
            positions[r][i] = std::sin(phase);
            positions[r][i + half] = std::cos(phase);
        }
    for (int il = 0; il < layers; ++il) {
        const auto p = "a.blk." + std::to_string(il) + ".";
        x = feed_forward(x, p, "");
        auto y = x;
        clip(y, -1e10f, 1e10f);
        y = attention(rms(y, p + "attn_pre_norm"), p, positions);
        clip(y, -1e10f, 1e10f);
        add(x, rms(y, p + "attn_post_norm"));
        // GGUF conv_norm is HF pre_layer_norm; norm_conv is HF conv_norm.
        y = clipped_linear(rms(x, p + "conv_norm"), p + "conv_pw1");
        HostMatrix gated(y.rows, hidden);
        for (int t = 0; t < y.rows; ++t)
            for (int d = 0; d < hidden; ++d)
                gated[t][d] = y[t][d] / (1.0f + std::exp(-y[t][d + hidden]));
        HostMatrix conv(y.rows, hidden);
        const auto *w = loader.weight(p + "conv_dw.weight");
        for (int t = 0; t < y.rows; ++t)
            for (int d = 0; d < hidden; ++d)
                for (int tap = 0; tap < 5; ++tap)
                    if (t + tap >= 4)
                        conv[t][d] += gated[t + tap - 4][d] * scalar(w, static_cast<size_t>(d) * 5 + tap);
        clip(conv, -1e10f, 1e10f);
        y = rms(conv, p + "norm_conv");
        silu(y);
        add(x, clipped_linear(y, p + "conv_pw2"));
        x = feed_forward(x, p, "_1");
        clip(x, -1e10f, 1e10f);
        x = rms(x, p + "ln2");
    }
    x = linear(x, "a.pre_encode.out", true);
    x = rms(x, loader.has_tensor_prefix("mm.a.soft_emb_norm.weight") ? "mm.a.soft_emb_norm" : "");
    x = linear(x, "mm.a.input_projection", false);
    finite(x);
    return x;
}
HostMatrix GemmaAudioEncoder::encode(const std::vector<float> &pcm) const { return encode_features(log_mel(pcm)); }
} // namespace clap::detail
