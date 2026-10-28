#include "models.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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
    if (!value.is_number_integer() || value <= 0 || value > maximum) invalid(field);
    return value.get<int>();
}
void fast_gelu(Matrix &x) {
    // HF gelu_pytorch_tanh / GGML GELU, rather than erf GELU in older encoders.
    for (float &v : x.data)
        v = 0.5f * v * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * v*v*v)));
}
}

EmbeddingGemma2Model::EmbeddingGemma2Model(const ModelLoader &source, int threads)
    : loader(source), ops(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
}

void EmbeddingGemma2Model::load_arch_hparams() {
    auto has = [&](const char *suffix) { return loader.has_metadata(key(suffix).c_str()); };
    auto value = [&](const char *suffix) { return loader.metadata_value(key(suffix).c_str()); };
    auto integer = [&](const char *suffix, int maximum = 65536) {
        return dimension(value(suffix), suffix, maximum);
    };
    auto optional = [&](const char *suffix, int fallback, int maximum = 65536) {
        return has(suffix) ? integer(suffix, maximum) : fallback;
    };
    auto number = [&](const char *suffix, double fallback) {
        if (!has(suffix)) return static_cast<float>(fallback);
        const auto v = value(suffix);
        if (!v.is_number()) invalid(suffix);
        const double n = v.get<double>();
        if (!std::isfinite(n) || n <= 0 || n > std::numeric_limits<float>::max() ||
            n < std::numeric_limits<float>::min()) invalid(suffix);
        return static_cast<float>(n);
    };
    hidden = integer("embedding_length");
    layers = integer("block_count", 1024);
    output = optional("embedding_length_out", hidden);
    per_layer = integer("embedding_length_per_layer_input");
    context = integer("context_length", 1048576);
    window = integer("attention.sliding_window", 1048576);
    if (!has("attention.layer_norm_rms_epsilon")) invalid("attention.layer_norm_rms_epsilon");
    epsilon = number("attention.layer_norm_rms_epsilon", 1e-6);
    rope_base = number("rope.freq_base", 10000);
    rope_base_swa = number("rope.freq_base_swa", 10000);
    if (has("attention.causal") && (!value("attention.causal").is_boolean() ||
                                     value("attention.causal").get<bool>())) invalid("causal attention");
    if (has("pooling_type") && (!value("pooling_type").is_number_integer() ||
                               value("pooling_type") != 1)) invalid("pooling_type (mean required)");
    if (has("attention.shared_kv_layers") && (!value("attention.shared_kv_layers").is_number_integer() ||
                                             value("attention.shared_kv_layers") != 0)) invalid("shared KV layers");
    if (has("rope.scaling.attn_factor") && number("rope.scaling.attn_factor", 1) != 1)
        invalid("rope.scaling.attn_factor");
    if (has("rope.scaling.type")) {
        const auto type = value("rope.scaling.type");
        if (type != "none" && type != "linear") invalid("rope.scaling.type");
        if (type == "linear") {
            if (!has("rope.scaling.factor")) invalid("rope.scaling.factor");
            rope_scale = 1.0f / number("rope.scaling.factor", 1);
        } else if (has("rope.scaling.factor") && number("rope.scaling.factor", 1) != 1) {
            invalid("rope.scaling.factor");
        }
    } else if (has("rope.scaling.factor")) {
        rope_scale = 1.0f / number("rope.scaling.factor", 1);
    }
    if (!std::isfinite(rope_scale) || rope_scale <= 0) invalid("rope frequency scale");
    const auto tokens = loader.metadata_value("tokenizer.ggml.tokens");
    if (!tokens.is_array() || tokens.empty() || tokens.size() > 1048576) invalid("vocabulary");
    vocab = static_cast<int>(tokens.size());
    auto special = [&](const char *name, int fallback) {
        if (!loader.has_metadata(name)) return fallback;
        const auto v = loader.metadata_value(name);
        if (!v.is_number_integer() || v < 0 || v >= vocab) invalid(name);
        return v.get<int>();
    };
    bos = special("tokenizer.ggml.bos_token_id", 2);
    eos = special("tokenizer.ggml.eos_token_id", 1);
    if (bos >= vocab || eos >= vocab) invalid("special token ids");
    pad = special("tokenizer.ggml.padding_token_id", 0);
    (void) special("tokenizer.ggml.unknown_token_id", 3);
    auto per_block = [&](const char *suffix, int fallback, bool required) {
        std::vector<int> result(layers, fallback);
        if (!has(suffix)) {
            if (required) invalid(suffix);
            return result;
        }
        const auto v = value(suffix);
        if (v.is_array()) {
            if (v.size() != static_cast<size_t>(layers)) invalid(suffix);
            for (int i = 0; i < layers; ++i) result[i] = dimension(v[i], suffix);
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
        if (pattern.size() != static_cast<size_t>(layers)) invalid("sliding window pattern length");
        for (int i = 0; i < layers; ++i) {
            if (pattern[i].is_boolean()) local[i] = pattern[i].get<bool>();
            else if (pattern[i].is_number_integer() && (pattern[i] == 0 || pattern[i] == 1))
                local[i] = pattern[i] == 1;
            else invalid("sliding window pattern element");
        }
    } else {
        const int pattern = optional("attention.sliding_window_pattern", 6, 1048576);
        for (int i = 0; i < layers; ++i) local[i] = (i + 1) % pattern != 0;
    }
    const int full_dimension = optional("attention.key_length", hidden / heads[0]);
    const int full_value = optional("attention.value_length", hidden / heads[0]);
    const int local_dimension = optional("attention.key_length_swa", full_dimension);
    const int local_value = optional("attention.value_length_swa", full_value);
    const int full_rotation = optional("rope.dimension_count", full_dimension);
    const int local_rotation = optional("rope.dimension_count_swa", full_rotation);
    if (full_dimension != full_value || local_dimension != local_value) invalid("key/value head dimension");
    dimensions.resize(layers); rotations.resize(layers);
    if (static_cast<int64_t>(per_layer) * layers > std::numeric_limits<int>::max() ||
        static_cast<int64_t>(hidden) * vocab > std::numeric_limits<int>::max()) invalid("derived tensor products");
    for (int i = 0; i < layers; ++i) {
        dimensions[i] = local[i] ? local_dimension : full_dimension;
        rotations[i] = local[i] ? local_rotation : full_rotation;
        if (dimensions[i] <= 0 || heads[i] % kv_heads[i] || rotations[i] % 2 ||
            rotations[i] > dimensions[i] || static_cast<int64_t>(heads[i]) * dimensions[i] > 1048576 ||
            static_cast<int64_t>(hidden) * ffn[i] > std::numeric_limits<int>::max()) invalid("attention/FFN geometry");
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
    throw std::runtime_error("CLAP: EmbeddingGemma2 is text-only; audio is unsupported");
}

Matrix EmbeddingGemma2Model::rms(const Matrix &x, const std::string &name, int width) const {
    if (!width) width = x.cols;
    Matrix out(x.rows, x.cols);
    const auto *weight = name.empty() ? nullptr : loader.weight(name + ".weight");
    // GGUF norm weights already contain HF's +1; never add it again.
    std::vector<float> scales(width, 1.0f);
    if (weight) tensor_row(weight, 0, scales.data());
    for (int row = 0; row < x.rows; ++row) for (int offset = 0; offset < x.cols; offset += width) {
        float square = 0;
        for (int c = 0; c < width; ++c) square += x[row][offset+c] * x[row][offset+c];
        const float scale = 1.0f / std::sqrt(square / width + epsilon);
        for (int c = 0; c < width; ++c) out[row][offset+c] = x[row][offset+c] * scale * scales[c];
    }
    return out;
}

Matrix EmbeddingGemma2Model::attention(const Matrix &x, int layer, const std::vector<int> &mask) const {
    const auto root = "blk." + std::to_string(layer) + ".";
    const int dim = dimensions[layer], rot = rotations[layer];
    const auto normalized = rms(x, root + "attn_norm");
    Matrix q = rms(ops.linear(normalized, root + "attn_q", false), root + "attn_q_norm", dim);
    Matrix k = rms(ops.linear(normalized, root + "attn_k", false), root + "attn_k_norm", dim);
    const Matrix v = rms(ops.linear(normalized, root + "attn_v", false), "", dim);
    const float base = local[layer] ? rope_base_swa : rope_base;
    const float frequency_scale = local[layer] ? 1.0f : rope_scale;
    // GGUF NEOX convention rotates the first n_rot channels; partial-RoPE
    // channels after n_rot remain untouched. Q/K checkpoint permutation is
    // performed by the GGUF converter, not here.
    for (Matrix *m : {&q, &k}) {
        const int count = m->cols / dim;
        for (int c = 0; c < rot / 2; ++c) {
            const float frequency = std::pow(base, -2.0f * c / rot) * frequency_scale;
            for (int row = 0; row < m->rows; ++row) {
                const float cosine = std::cos(row * frequency), sine = std::sin(row * frequency);
                for (int head = 0; head < count; ++head) {
                    float *p = (*m)[row] + head * dim;
                    const float a = p[c], b = p[c + rot/2];
                    p[c] = a*cosine - b*sine; p[c + rot/2] = a*sine + b*cosine;
                }
            }
        }
    }
    Matrix result(x.rows, q.cols);
    // One score row at a time avoids a context^2 attention allocation.
    std::vector<float> scores(x.rows);
    for (int row = 0; row < x.rows; ++row) {
        // GGUF stores twice the HF radius. Symmetric SWA includes both
        // endpoints: abs(query_position - key_position) <= window / 2.
        const int begin = local[layer] ? std::max(0, row - window / 2) : 0;
        const int end = local[layer] ? static_cast<int>(std::min<int64_t>(x.rows, static_cast<int64_t>(row) + window / 2 + 1)) : x.rows;
        for (int head = 0; head < heads[layer]; ++head) {
            const int kv_head = head / (heads[layer] / kv_heads[layer]);
            float maximum = -std::numeric_limits<float>::infinity();
            for (int pos = begin; pos < end; ++pos) {
                if (!mask[pos]) continue;
                float dot = 0;
                for (int c = 0; c < dim; ++c) dot += q[row][head*dim+c] * k[pos][kv_head*dim+c];
                // Unit attention scale: Q and K are already learned RMS-normalized.
                scores[pos] = dot; maximum = std::max(maximum, dot);
            }
            float total = 0;
            for (int pos = begin; pos < end; ++pos)
                if (mask[pos]) total += (scores[pos] = std::exp(scores[pos] - maximum));
            for (int pos = begin; pos < end; ++pos) {
                if (!mask[pos]) continue;
                const float probability = scores[pos] / total;
                for (int c = 0; c < dim; ++c) result[row][head*dim+c] += probability * v[pos][kv_head*dim+c];
            }
        }
    }
    return rms(ops.linear(result, root + "attn_output", false), root + "post_attention_norm");
}

std::vector<float> EmbeddingGemma2Model::text(const std::vector<int> &ids) const {
    if (ids.empty() || ids.size() > static_cast<size_t>(context)) invalid("token sequence length");
    for (int id : ids) if (id < 0 || id >= vocab) invalid("token id");
    std::vector<int> mask(ids.size());
    int real_tokens = 0;
    for (size_t i = 0; i < ids.size(); ++i) real_tokens += (mask[i] = ids[i] != pad);
    if (!real_tokens) invalid("all-padding token input");
    int64_t width = std::max<int64_t>(std::max(hidden, output), static_cast<int64_t>(per_layer) * layers);
    for (int i = 0; i < layers; ++i) width = std::max<int64_t>(width, std::max(heads[i] * dimensions[i], ffn[i]));
    if (ids.size() > static_cast<size_t>(std::numeric_limits<int>::max() / width)) invalid("token matrix size");
    Matrix x(static_cast<int>(ids.size()), hidden);
    const auto *embedding = loader.weight("token_embd.weight");
    const float scale = std::sqrt(static_cast<float>(hidden));
    for (int row = 0; row < x.rows; ++row) {
        tensor_row(embedding, ids[row], x[row]);
        for (int c = 0; c < hidden; ++c) x[row][c] *= scale;
    }
    Matrix projected = ops.linear(x, "per_layer_model_proj", false);
    for (float &v : projected.data) v /= scale;
    projected = rms(projected, "per_layer_proj_norm", per_layer);
    for (int layer = 0; layer < layers; ++layer) {
        const auto root = "blk." + std::to_string(layer) + ".";
        add(x, attention(x, layer, mask));
        const auto pre = rms(x, root + "ffn_norm");
        Matrix gate = ops.linear(pre, root + "ffn_gate", false);
        fast_gelu(gate);
        const Matrix up = ops.linear(pre, root + "ffn_up", false);
        for (size_t i = 0; i < gate.data.size(); ++i) gate.data[i] *= up.data[i];
        add(x, rms(ops.linear(gate, root + "ffn_down", false), root + "post_ffw_norm"));
        Matrix ple = ops.linear(x, root + "inp_gate", false);
        fast_gelu(ple);
        for (int row = 0; row < x.rows; ++row) for (int c = 0; c < per_layer; ++c)
            ple[row][c] *= projected[row][layer*per_layer+c];
        add(x, rms(ops.linear(ple, root + "proj", false), root + "post_norm"));
        const float out_scale = scalar(loader.weight(root + "layer_output_scale.weight"), 0);
        for (float &v : x.data) v *= out_scale;
    }
    // Match the reference graph's operation order even for storage types that
    // pack/round the activation operand: projecting a pooled row first is only
    // algebraically equivalent, not equivalent under F16/Q8 input packing.
    const Matrix final = ops.linear(rms(x, "output_norm"), "output", false);
    std::vector<float> result(output, 0.0f);
    for (int row = 0; row < final.rows; ++row) if (mask[row]) for (int c = 0; c < output; ++c)
        result[c] += final[row][c] / real_tokens;
    double square = 0;
    for (float v : result) {
        if (!std::isfinite(v)) invalid("non-finite output embedding");
        square += static_cast<double>(v) * v;
    }
    if (!(square > 0)) invalid("zero output embedding");
    const float inverse = static_cast<float>(1.0 / std::sqrt(square));
    for (float &v : result) v *= inverse;
    return result;
}
} // namespace clap::detail
