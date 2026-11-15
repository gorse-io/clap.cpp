/* Copyright 2026 clap.cpp Authors. Licensed under Apache-2.0. */
#include "audio.h"
#include "context.h"
#include "voiceclap.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <regex>
#include <stdexcept>

namespace clap::detail {
namespace {
int dimension(const nlohmann::json &j, const char *key, int limit = 1048576) {
    const auto &v = j.at(key);
    if (!v.is_number_integer() || v.get<int64_t>() <= 0 || v.get<int64_t>() > limit)
        throw std::runtime_error(std::string("CLAP: invalid VoiceCLAP dimension: ") + key);
    return v.get<int>();
}
void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(std::string("CLAP: VoiceCLAP ") + message);
}
ggml_tensor *rms(Graph &g, ggml_tensor *x, const LayerWeights &norm, float epsilon) {
    return ggml_mul(g.ctx(), ggml_rms_norm(g.ctx(), x, epsilon), g.weight(norm.weight));
}
ggml_tensor *token_slice(Graph &g, ggml_tensor *x, int64_t start, int64_t count) {
    return ggml_cont(g.ctx(), ggml_view_2d(g.ctx(), x, x->ne[0], count, x->nb[1], start * x->nb[1]));
}
// Repeat each KV head consecutively (not modulo head_count). Qwen's
// repeat_kv expands [kv, repeats, tokens, head_dim] before flattening heads.
ggml_tensor *repeat_kv(Graph &g, ggml_tensor *x, int dim, int kv_heads, int heads) {
    auto *ctx = g.ctx();
    const auto tokens = x->ne[1];
    x = ggml_permute(ctx, ggml_reshape_3d(ctx, x, dim, kv_heads, tokens), 0, 2, 1, 3);
    x = ggml_reshape_3d(ctx, ggml_cont(ctx, x), dim * tokens, 1, kv_heads);
    x = ggml_repeat_4d(ctx, x, dim * tokens, heads / kv_heads, kv_heads, 1);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, x), dim, tokens, heads);
}
} // namespace

bool VoiceClapModel::tensor_quantizable(const ggml_tensor *) {
    // Only independently verified floating storage is supported for this model.
    return false;
}
TensorLoadPolicy VoiceClapModel::loading_policy(const ModelLoader &) {
    auto policy = encoder_loading_policy([](const ggml_tensor *w) {
        require(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16, "supports only F32/F16 tensor storage");
    });
    policy.unrecognized_tensor_error = "CLAP: unrecognized VoiceCLAP tensor: ";
    policy.validate_tensor_contract = [](const ggml_tensor *w, std::initializer_list<int64_t> dims) {
        if (dims.size() == 1 && w->type != GGML_TYPE_F32)
            throw std::runtime_error("CLAP: VoiceCLAP normalization/bias vectors require F32: " + std::string(w->name));
    };
    return policy;
}
VoiceClapModel::VoiceClapModel(const ModelLoader &source) : loader(source) {
    load_arch_hparams();
    load_arch_tensors();
}
void VoiceClapModel::load_arch_hparams() {
    config = nlohmann::json::parse(loader.metadata("voiceclap.config"));
    preprocessor = nlohmann::json::parse(loader.metadata("voiceclap.preprocessor"));
    require(config.at("model_type") == "qwen2_5_omni_thinker" &&
                config.at("architectures") == nlohmann::json::array({"Qwen2_5OmniThinkerForConditionalGeneration"}),
            "requires Qwen2.5-Omni Thinker config");
    const auto &t = config.at("text_config"), &a = config.at("audio_config");
    require(t.at("model_type") == "qwen2_5_omni_text" && a.at("model_type") == "qwen2_5_omni_audio_encoder",
            "unsupported encoder/decoder model type");
    h.hidden = dimension(t, "hidden_size");
    h.heads = dimension(t, "num_attention_heads");
    h.kv_heads = dimension(t, "num_key_value_heads");
    h.intermediate = dimension(t, "intermediate_size");
    h.vocab = dimension(t, "vocab_size");
    h.context = dimension(t, "max_position_embeddings");
    h.text_layers = dimension(t, "num_hidden_layers", 1000);
    h.head_dim = t.contains("head_dim") ? dimension(t, "head_dim") : h.hidden / h.heads;
    require(h.head_dim > 0 && h.head_dim % 2 == 0 && h.hidden == h.heads * int64_t(h.head_dim) &&
                h.heads % h.kv_heads == 0,
            "invalid decoder GQA/head dimensions");
    h.epsilon = t.at("rms_norm_eps");
    h.theta = t.at("rope_theta");
    require(std::isfinite(h.epsilon) && h.epsilon > 0 && std::isfinite(h.theta) && h.theta > 0,
            "invalid RMS/RoPE constants");
    require(t.at("hidden_act") == "silu" && !t.value("use_sliding_window", false) && !t.value("mlp_bias", false) &&
                t.value("attention_bias", true),
            "unsupported decoder activation/attention variant");
    const auto &rope = t.at("rope_scaling");
    require(rope.value("rope_type", std::string("default")) == "default" &&
                rope.value("type", std::string("default")) == "default",
            "unsupported RoPE scaling");
    const auto &sections = rope.at("mrope_section");
    require(sections.is_array() && sections.size() == 3, "invalid mRoPE sections");
    int64_t sum = 0;
    for (const auto &section : sections) {
        require(section.is_number_integer() && section.get<int64_t>() > 0 && section.get<int64_t>() <= h.head_dim,
                "invalid mRoPE section");
        sum += section.get<int64_t>();
    }
    require(sum == h.head_dim / 2, "mRoPE sections do not cover head dimension");
    if (t.contains("layer_types")) {
        require(t.at("layer_types").size() == size_t(h.text_layers), "invalid decoder layer_types length");
        for (const auto &type : t.at("layer_types"))
            require(type == "full_attention", "requires full causal attention");
    }
    h.audio_hidden = dimension(a, "d_model");
    h.audio_heads = dimension(a, "encoder_attention_heads");
    h.audio_intermediate = dimension(a, "encoder_ffn_dim");
    h.audio_layers = dimension(a, "encoder_layers", 1000);
    h.mels = dimension(a, "num_mel_bins", 1024);
    h.window = dimension(a, "n_window", 10000);
    h.audio_positions = dimension(a, "max_source_positions");
    require(h.mels == 128 && h.window == 100, "unsupported audio mel/chunk variant");
    require(h.audio_hidden >= 4 && h.audio_hidden % 2 == 0 && h.audio_hidden % h.audio_heads == 0 &&
                h.audio_positions >= h.window && dimension(a, "output_dim") == h.hidden &&
                a.at("activation_function") == "gelu" && !a.value("scale_embedding", false),
            "unsupported audio dimensions/activation/scaling");
    require(!a.contains("num_hidden_layers") || a.at("num_hidden_layers") == h.audio_layers,
            "inconsistent audio layer counts");
    require(preprocessor.at("feature_extractor_type") == "WhisperFeatureExtractor" &&
                dimension(preprocessor, "feature_size") == h.mels &&
                dimension(preprocessor, "sampling_rate") == 16000 && dimension(preprocessor, "n_fft") == 400 &&
                dimension(preprocessor, "hop_length") == 160 && preprocessor.value("padding_value", 0.0) == 0.0 &&
                preprocessor.value("padding_side", std::string("right")) == "right" &&
                preprocessor.value("dither", 0.0) == 0.0,
            "unsupported Whisper frontend");
    h.max_samples = dimension(preprocessor, "n_samples", 16000 * 3600);
    require(h.max_samples >= 400 && h.max_samples % 160 == 0 &&
                (!preprocessor.contains("chunk_length") ||
                 dimension(preprocessor, "chunk_length", 3600) * int64_t(16000) == h.max_samples) &&
                (!preprocessor.contains("nb_max_frames") ||
                 dimension(preprocessor, "nb_max_frames") == h.max_samples / 160),
            "inconsistent Whisper duration");
    auto special_id = [&](const char *key) {
        const auto &id = config.at(key);
        require(id.is_number_integer() && id.get<int64_t>() >= 0 && id.get<int64_t>() < h.vocab,
                "special token outside vocabulary");
        return id.get<int>();
    };
    h.bos = special_id("bos_token_id");
    h.eos = special_id("eos_token_id");
    h.audio_bos = special_id("audio_start_token_id");
    h.audio_eos = special_id("audio_end_token_id");
    h.audio_token = special_id("audio_token_index");
    auto template_ids = [&](const char *key) {
        const auto values = nlohmann::json::parse(loader.metadata(key));
        require(values.is_array(), "audio template IDs must be arrays");
        std::vector<int> ids;
        for (const auto &id : values) {
            require(id.is_number_integer() && id.get<int64_t>() >= 0 && id.get<int64_t>() < h.vocab,
                    "invalid audio template token ID");
            ids.push_back(id.get<int>());
        }
        validate_ids(ids);
        return ids;
    };
    audio_prefix = template_ids("voiceclap.audio_prefix");
    audio_suffix = template_ids("voiceclap.audio_suffix");
    const auto pooling = nlohmann::json::parse(loader.metadata("voiceclap.pooling"));
    require(pooling.at("pooling_mode") == "lasttoken" && pooling.at("include_prompt") == true &&
                pooling.at("embedding_dimension") == h.hidden,
            "unsupported pooling configuration");
    const auto sentence = nlohmann::json::parse(loader.metadata("voiceclap.sentence_bert"));
    require(sentence.at("transformer_task") == "any-to-any" &&
                sentence.at("module_output_name") == "token_embeddings" &&
                sentence.at("processing_kwargs").at("chat_template") ==
                    nlohmann::json({{"chat_template", "sentence_transformers"}, {"add_generation_prompt", true}}),
            "unsupported sentence-transformer output/template configuration");
    for (const auto *modality : {"text", "audio", "message"}) {
        const auto &m = sentence.at("modality_config").at(modality);
        require(m.at("method") == "forward" &&
                    m.at("method_output_name") == nlohmann::json::array({"hidden_states", -1}),
                "unsupported sentence-transformer modality configuration");
    }
    require(sentence.at("modality_config").at("message").at("format") == "structured",
            "requires structured sentence-transformer messages");
    require(audio_prefix.back() == h.audio_bos && audio_suffix.front() == h.audio_eos,
            "audio template must delimit projected tokens with audio BOS/EOS");
}
void VoiceClapModel::load_arch_tensors() {
    token_embedding = loader.create_tensor("text.embed_tokens.weight", {h.hidden, h.vocab});
    final_norm = create_layer(loader, "text.norm", {h.hidden});
    for (int i = 0; i < h.text_layers; ++i) {
        const auto p = "text.layers." + std::to_string(i);
        DecoderLayer l;
        l.attention.query = create_linear(loader, p + ".attn.q_proj", h.hidden, h.hidden);
        l.attention.key = create_linear(loader, p + ".attn.k_proj", h.hidden, h.kv_heads * h.head_dim);
        l.attention.value = create_linear(loader, p + ".attn.v_proj", h.hidden, h.kv_heads * h.head_dim);
        l.attention.output = create_linear(loader, p + ".attn.o_proj", h.hidden, h.hidden, false);
        l.attention_norm = create_layer(loader, p + ".attn_norm", {h.hidden});
        l.ffn_norm = create_layer(loader, p + ".ffn_norm", {h.hidden});
        l.gate = create_linear(loader, p + ".mlp.gate_proj", h.hidden, h.intermediate, false);
        l.up = create_linear(loader, p + ".mlp.up_proj", h.hidden, h.intermediate, false);
        l.down = create_linear(loader, p + ".mlp.down_proj", h.intermediate, h.hidden, false);
        decoder_layers.push_back(l);
    }
    conv1 = create_layer(loader, "audio.conv1", {3, h.mels, h.audio_hidden}, {h.audio_hidden});
    conv2 = create_layer(loader, "audio.conv2", {3, h.audio_hidden, h.audio_hidden}, {h.audio_hidden});
    for (int i = 0; i < h.audio_layers; ++i) {
        const auto p = "audio.layers." + std::to_string(i);
        TransformerWeights l;
        l.attention.query = create_linear(loader, p + ".attn.q_proj", h.audio_hidden, h.audio_hidden);
        l.attention.key = create_linear(loader, p + ".attn.k_proj", h.audio_hidden, h.audio_hidden, false);
        l.attention.value = create_linear(loader, p + ".attn.v_proj", h.audio_hidden, h.audio_hidden);
        l.attention.output = create_linear(loader, p + ".attn.out_proj", h.audio_hidden, h.audio_hidden);
        l.attention_norm = create_norm(loader, p + ".attn_norm", h.audio_hidden);
        l.output_norm = create_norm(loader, p + ".ffn_norm", h.audio_hidden);
        l.intermediate = create_linear(loader, p + ".fc1", h.audio_hidden, h.audio_intermediate);
        l.output = create_linear(loader, p + ".fc2", h.audio_intermediate, h.audio_hidden);
        encoder_layers.push_back(l);
    }
    audio_norm = create_norm(loader, "audio.ln_post", h.audio_hidden);
    audio_projection = create_linear(loader, "audio.proj", h.audio_hidden, h.hidden);
    // This trained but unused embedding belongs to the official audio tower.
    loader.create_tensor("audio.audio_bos_eos_token.weight", {h.hidden, 2}, TensorRequirement::Optional);
}
void VoiceClapModel::initialize_context(Context &context) const { context.runtime(this, loader, false); }
TextParameters VoiceClapModel::text_parameters() const { return {-1, -1, size_t(h.context)}; }
void VoiceClapModel::validate_ids(const std::vector<int> &ids) const {
    require(!ids.empty() && ids.size() <= size_t(h.context), "empty or over-context token sequence");
    for (int id : ids)
        require(id >= 0 && id < h.vocab, "token ID outside vocabulary");
}
ggml_tensor *VoiceClapModel::decode(Graph &g, ggml_tensor *x) const {
    auto *ctx = g.ctx();
    const auto n = x->ne[1];
    require(n > 0 && n <= h.context, "decoder sequence exceeds context");
    for (const auto &l : decoder_layers) {
        auto *z = rms(g, x, l.attention_norm, h.epsilon);
        // Text-only and audio-only Omni positions are identical on T/H/W.
        // Consequently all mRoPE sections reduce exactly to half-split RoPE.
        auto *q = g.rope(g.linear(z, l.attention.query), h.heads, h.theta);
        auto *k = g.rope(g.linear(z, l.attention.key), h.kv_heads, h.theta);
        auto *v = g.linear(z, l.attention.value);
        q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, h.head_dim, h.heads, n), 0, 2, 1, 3);
        k = repeat_kv(g, k, h.head_dim, h.kv_heads, h.heads);
        v = repeat_kv(g, v, h.head_dim, h.kv_heads, h.heads);
        auto *scores = ggml_scale(ctx, g.matmul(k, q), 1.0f / std::sqrt(float(h.head_dim)));
        scores = ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, scores, 0));
        auto *y = g.matmul(ggml_cont(ctx, ggml_transpose(ctx, v)), scores);
        y = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3)), h.hidden, n);
        x = ggml_add(ctx, x, g.linear(y, l.attention.output));
        z = rms(g, x, l.ffn_norm, h.epsilon);
        y = ggml_mul(ctx, ggml_silu(ctx, g.linear(z, l.gate)), g.linear(z, l.up));
        x = ggml_add(ctx, x, g.linear(y, l.down));
    }
    x = rms(g, x, final_norm, h.epsilon);
    return g.l2_normalize(token_slice(g, x, n - 1, 1));
}
std::vector<float> VoiceClapModel::text(Context &context, const std::vector<int> &ids) const {
    validate_ids(ids);
    auto result =
        context.runtime(this, loader, false)
            .run_cached(
                "voiceclap.text", {GraphInput(ids)},
                [&](Graph &g) {
                    return std::vector<ggml_tensor *>{decode(g, g.embedding(token_embedding, g.dynamic_input(0)))};
                },
                size_t(h.text_layers) * 180);
    return std::move(result.front().values);
}
ggml_tensor *VoiceClapModel::audio_chunk(Graph &g, ggml_tensor *features) const {
    auto *ctx = g.ctx();
    // Processing the unpadded tail is equivalent to masking conv1 immediately
    // before conv2: its implicit right zero pad is the upstream padded mask.
    auto *x = ggml_gelu_erf(ctx, g.conv1d(features, conv1, 3, 1, 1, 1));
    x = ggml_gelu_erf(ctx, g.conv1d(x, conv2, 3, 2, 1, 1));
    std::vector<float> position(size_t(x->ne[1]) * h.audio_hidden);
    for (int64_t t = 0; t < x->ne[1]; ++t) {
        for (int d = 0; d < h.audio_hidden / 2; ++d) {
            const float inv = std::exp(-float(std::log(10000.0) / (h.audio_hidden / 2 - 1)) * float(d));
            const float phase = float(t) * inv;
            position[size_t(t) * h.audio_hidden + d] = std::sin(phase);
            position[size_t(t) * h.audio_hidden + d + h.audio_hidden / 2] = std::cos(phase);
        }
    }
    x = ggml_add(ctx, x, g.input(position, h.audio_hidden, x->ne[1]));
    for (const auto &l : encoder_layers) {
        auto *z = g.norm(x, l.attention_norm, 1e-5f);
        auto *q = g.linear(z, l.attention.query), *k = g.linear(z, l.attention.key),
             *v = g.linear(z, l.attention.value);
        x = ggml_add(ctx, x, g.linear(g.attention(q, k, v, h.audio_heads), l.attention.output));
        z = g.norm(x, l.output_norm, 1e-5f);
        x = ggml_add(ctx, x, g.linear(ggml_gelu_erf(ctx, g.linear(z, l.intermediate)), l.output));
    }
    return x;
}
HostMatrix VoiceClapModel::audio_tokens(Context &context, const HostMatrix &features) const {
    require(features.cols == h.mels && features.rows >= 3 && features.rows <= h.max_samples / 160,
            "invalid audio feature shape");
    auto &runtime = context.runtime(this, loader, false);
    std::vector<float> hidden;
    for (int offset = 0; offset < features.rows; offset += 2 * h.window) {
        const int count = std::min(2 * h.window, features.rows - offset);
        std::vector<float> chunk(features.data.begin() + size_t(offset) * h.mels,
                                 features.data.begin() + size_t(offset + count) * h.mels);
        auto out = runtime.run_cached(
            "voiceclap.audio.chunk", {GraphInput(chunk, h.mels, count)},
            [&](Graph &g) { return std::vector<ggml_tensor *>{audio_chunk(g, g.dynamic_input(0))}; },
            size_t(h.audio_layers) * 150);
        hidden.insert(hidden.end(), out.front().values.begin(), out.front().values.end());
    }
    const int frames = int(hidden.size() / h.audio_hidden), tokens = frames / 2;
    require(tokens > 0, "audio too short for average pooling");
    auto out =
        runtime.run_cached("voiceclap.audio.project", {GraphInput(hidden, h.audio_hidden, frames)}, [&](Graph &g) {
            std::vector<int> even(tokens), odd(tokens);
            for (int i = 0; i < tokens; ++i) {
                even[i] = 2 * i;
                odd[i] = 2 * i + 1;
            }
            auto *x = g.dynamic_input(0);
            auto *a = ggml_get_rows(g.ctx(), x, g.indices(even));
            auto *b = ggml_get_rows(g.ctx(), x, g.indices(odd));
            x = ggml_scale(g.ctx(), ggml_add(g.ctx(), a, b), 0.5f);
            return std::vector<ggml_tensor *>{g.linear(g.norm(x, audio_norm, 1e-5f), audio_projection)};
        });
    HostMatrix result(tokens, h.hidden);
    result.data = std::move(out.front().values);
    return result;
}
std::vector<float> VoiceClapModel::audio_features_embedding(Context &context, const HostMatrix &features) const {
    auto tokens = audio_tokens(context, features);
    require(audio_prefix.size() + size_t(tokens.rows) + audio_suffix.size() <= size_t(h.context),
            "audio plus template exceeds decoder context");
    auto out = context.runtime(this, loader, false)
                   .run_cached(
                       "voiceclap.audio.decode", {GraphInput(tokens)},
                       [&](Graph &g) {
                           auto *x =
                               ggml_concat(g.ctx(), g.embedding(token_embedding, audio_prefix), g.dynamic_input(0), 1);
                           x = ggml_concat(g.ctx(), x, g.embedding(token_embedding, audio_suffix), 1);
                           return std::vector<ggml_tensor *>{decode(g, x)};
                       },
                       size_t(h.text_layers) * 180);
    return std::move(out.front().values);
}
AudioFrames VoiceClapModel::audio_frames(Context &, const std::vector<float> &) const {
    throw std::runtime_error("CLAP: VoiceCLAP exposes joint embeddings, not waveform frames");
}

// Whisper's centered 400-point STFT using Bluestein and the shared native
// radix-2 FFT. No Python, external frontend, or inference subprocess is used.
HostMatrix VoiceClapModel::frontend(const std::vector<float> &pcm, int mels, int max_samples) {
    require(!pcm.empty() && pcm.size() <= (size_t(1) << 28) &&
                std::all_of(pcm.begin(), pcm.end(), [](float v) { return std::isfinite(v); }),
            "requires a finite nonempty waveform of at most 1 GiB");
    constexpr int n = 400, hop = 160, bins = 201, size = 1024;
    constexpr double pi = 3.14159265358979323846264338327950288;
    const size_t samples = std::min(pcm.size(), size_t(max_samples));
    const int frames = int((samples + hop - 1) / hop);
    require(frames >= 3, "audio too short for encoder pooling");
    // The processor pads to max_length BEFORE extracting mel features, and
    // normalization uses the maximum over the entire padded spectrogram.
    // Only up to two extra frames can contain nonzero STFT contributions.
    const int compute_frames = std::min(max_samples / hop, int((samples + 199) / hop + 1));
    std::vector<float> mel(size_t(compute_frames) * mels);
    std::vector<double> filters(size_t(mels) * bins), window(n);
    auto hz_to_mel = [](double hz) {
        return hz < 1000 ? hz / (200.0 / 3) : 15.0 + std::log(hz / 1000.0) / (std::log(6.4) / 27.0);
    };
    auto mel_to_hz = [](double m) {
        return m < 15 ? m * (200.0 / 3) : 1000.0 * std::exp((m - 15.0) * (std::log(6.4) / 27.0));
    };
    std::vector<double> points(mels + 2);
    for (int i = 0; i < mels + 2; ++i)
        points[i] = mel_to_hz(hz_to_mel(8000) * i / (mels + 1));
    for (int m = 0; m < mels; ++m)
        for (int f = 0; f < bins; ++f) {
            const double hz = double(f) * 16000 / n;
            filters[size_t(m) * bins + f] =
                std::max(0.0, std::min((hz - points[m]) / (points[m + 1] - points[m]),
                                       (points[m + 2] - hz) / (points[m + 2] - points[m + 1]))) *
                2 / (points[m + 2] - points[m]);
        }
    std::vector<std::complex<double>> chirp(n), kernel(size), work(size);
    for (int i = 0; i < n; ++i) {
        window[i] = 0.5f - 0.5f * std::cos(float(2 * pi / n) * float(i));
        chirp[i] = std::polar(1.0, -pi * double(i) * i / n);
        kernel[i] = std::conj(chirp[i]);
        if (i)
            kernel[size - i] = kernel[i];
    }
    clap::fft(kernel);
    auto sample = [&](int64_t at) -> double {
        // Reflection belongs to the MAX-LENGTH padded waveform, not the
        // short raw recording. Its right tail is zero for short inputs.
        if (at < 0)
            at = -at;
        if (at >= max_samples)
            at = 2 * int64_t(max_samples) - 2 - at;
        return at >= 0 && size_t(at) < samples ? pcm[size_t(at)] : 0.0;
    };
    float maximum = -10.0f;
    std::vector<float> power(bins);
    for (int t = 0; t < compute_frames; ++t) {
        std::fill(work.begin(), work.end(), std::complex<double>{});
        for (int j = 0; j < n; ++j)
            work[j] = double(float(sample(int64_t(t) * hop + j - n / 2) * window[j])) * chirp[j];
        clap::fft(work);
        for (int j = 0; j < size; ++j)
            work[j] = std::conj(work[j] * kernel[j]);
        clap::fft(work);
        for (int f = 0; f < bins; ++f)
            power[f] = float(std::norm(std::conj(work[f]) / double(size) * chirp[f]));
        for (int m = 0; m < mels; ++m) {
            double sum = 0;
            for (int f = 0; f < bins; ++f)
                sum += power[f] * filters[size_t(m) * bins + f];
            const float value = float(std::log10(std::max(sum, 1e-10)));
            mel[size_t(t) * mels + m] = value;
            maximum = std::max(maximum, value);
        }
    }
    HostMatrix result(frames, mels);
    for (size_t i = 0; i < result.data.size(); ++i)
        result.data[i] = (std::max(mel[i], maximum - 8.0f) + 4.0f) / 4.0f;
    return result;
}
std::vector<float> VoiceClapModel::waveform_embedding(Context &context, const std::vector<float> &pcm,
                                                      size_t crop_offset) const {
    require(crop_offset == 0, "crop offsets are unsupported");
    return audio_features_embedding(context, frontend(pcm, h.mels, h.max_samples));
}
} // namespace clap::detail
