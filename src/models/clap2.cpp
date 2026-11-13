#include "audio.h"
#include "context.h"
#include "models.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace clap::detail {
TensorLoadPolicy Clap2Model::loading_policy(const ModelLoader &) {
    return encoder_loading_policy([](const ggml_tensor *w) {
        if (ggml_is_quantized(w->type))
            throw std::runtime_error("CLAP: CLAP2 block-quantized weights are not supported; use F32/F16/BF16");
    });
}

Clap2Model::Clap2Model(const ModelLoader &source) : loader(source) {
    load_arch_hparams();
    load_arch_tensors();
}
void Clap2Model::initialize_context(Context &context) const { context.runtime(this, loader, true); }

void Clap2Model::load_arch_hparams() {
    const auto config = nlohmann::json::parse(loader.metadata("clap2.config"));
    const auto processor = nlohmann::json::parse(loader.metadata("clap2.processor_config"));
    const auto audio = config.at("audio_config");
    const auto txt = config.at("text_config");
    auto &bins = h.bins, &patch = h.patch, &rate = h.rate, &nfft = h.nfft, &hop = h.hop;
    auto &fmin = h.fmin, &fmax = h.fmax;
    auto dim = [](const nlohmann::json &j, const char *key, int maximum = 65536) {
        const auto &v = j.at(key);
        if (!v.is_number_integer() || v <= 0 || v > maximum)
            throw std::runtime_error(std::string("CLAP: invalid CLAP2 dimension: ") + key);
        return v.get<int>();
    };
    dim(config, "projection_dim");
    for (const auto *j : {&audio, &txt}) {
        int width = dim(*j, "hidden_size"), heads = dim(*j, "num_attention_heads");
        dim(*j, "num_hidden_layers", 256);
        int intermediate = dim(*j, "intermediate_size");
        if (width % heads || (width / heads) % 4 || int64_t(width) * intermediate * 2 > INT32_MAX ||
            j->at("hidden_act") != "silu" || !j->at("rms_norm_eps").is_number() ||
            !std::isfinite(j->at("rms_norm_eps").get<float>()) || j->at("rms_norm_eps").get<float>() <= 0 ||
            !j->at("rope_parameters").at("rope_theta").is_number() ||
            !std::isfinite(j->at("rope_parameters").at("rope_theta").get<float>()) ||
            j->at("rope_parameters").at("rope_theta").get<float>() <= 0)
            throw std::runtime_error("CLAP: unsupported CLAP2 transformer geometry/activation");
    }
    dim(txt, "vocab_size", INT32_MAX);
    dim(txt, "max_position_embeddings");
    bins = dim(audio, "num_mel_bins", 2048);
    patch = dim(audio, "patch_size", 256);
    if (bins % patch || dim(audio, "num_register_tokens", 256) != 1 ||
        audio.at("rope_parameters").at("rope_type") != "axial" ||
        txt.at("rope_parameters").at("rope_type") != "default" || audio.at("use_qk_norm") != false ||
        txt.at("use_qk_norm") != true || audio.at("attention_bias") != true || audio.at("mlp_bias") != true ||
        txt.at("attention_bias") != false || txt.at("mlp_bias") != false || audio.at("pooler_hidden_act") != "gelu" ||
        config.at("projection_hidden_act") != "gelu" || !txt.at("layerscale_value").is_null())
        throw std::runtime_error("CLAP: unsupported CLAP2 architecture variant");
    const auto &f = processor.at("feature_extractor");
    if (!f.contains("max_num_patches") || !f.at("max_num_patches").is_null())
        throw std::runtime_error("CLAP: CLAP2 requires unlimited max_num_patches");
    rate = dim(f, "sampling_rate", 384000);
    nfft = dim(f, "n_fft", 16384);
    hop = dim(f, "hop_length", 16384);
    fmin = f.at("f_min");
    fmax = f.at("f_max");
    if ((nfft & (nfft - 1)) || hop > nfft || f.at("feature_size") != bins || f.at("patch_size") != patch ||
        !std::isfinite(fmin) || !std::isfinite(fmax) || fmin < 0 || fmax <= fmin || fmax > rate / 2)
        throw std::runtime_error("CLAP: invalid CLAP2 frontend");
    const auto tokenizer = nlohmann::json::parse(loader.metadata("clap2.tokenizer"));
    const auto &tm = tokenizer.at("model"), &pre = tokenizer.at("pre_tokenizer"),
               &post = tokenizer.at("post_processor");
    const auto expected_template = nlohmann::json::parse(
        R"([{"SpecialToken":{"id":"<|bos|>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<|eos|>","type_id":0}}])");
    if (tm.at("type") != "BPE" || !tokenizer.at("normalizer").is_null() || pre.at("type") != "ByteLevel" ||
        pre.at("add_prefix_space") != false || pre.at("use_regex") != true ||
        tokenizer.at("decoder").at("type") != "ByteLevel" || post.at("type") != "TemplateProcessing" ||
        post.at("single") != expected_template)
        throw std::runtime_error("CLAP: unsupported CLAP2 tokenizer/template");
    for (const char *key : {"bos", "eos"}) {
        const std::string token = std::string("<|") + key + "|>";
        const auto &special = post.at("special_tokens").at(token);
        if (special.at("ids") != nlohmann::json::array({txt.at(std::string(key) + "_token_id")}) ||
            special.at("tokens") != nlohmann::json::array({token}))
            throw std::runtime_error("CLAP: invalid CLAP2 tokenizer special token");
    }
    for (const char *id : {"bos_token_id", "eos_token_id", "pad_token_id"}) {
        const auto &v = txt.at(id);
        if (!v.is_number_integer() || v < 0 || v >= txt.at("vocab_size"))
            throw std::runtime_error("CLAP: invalid CLAP2 special token");
    }
    h.projection = config.at("projection_dim");
    h.vocab = txt.at("vocab_size");
    h.positions = txt.at("max_position_embeddings");
    h.bos = txt.at("bos_token_id");
    h.eos = txt.at("eos_token_id");
    h.pad = txt.at("pad_token_id");
    auto parameters = [](const nlohmann::json &j) -> BranchParameters {
        return {j.at("hidden_size"),       j.at("num_attention_heads"), j.at("intermediate_size"),
                j.at("num_hidden_layers"), j.at("rms_norm_eps"),        j.at("rope_parameters").at("rope_theta")};
    };
    audio_branch.h = parameters(audio);
    text_branch.h = parameters(txt);
}
void Clap2Model::load_arch_tensors() {
    for (bool is_audio : {true, false}) {
        auto &branch = is_audio ? audio_branch : text_branch;
        const auto &parameters = branch.h;
        const std::string root = is_audio ? "audio_model" : "text_model";
        const int w = parameters.hidden, m = parameters.intermediate, d = w / parameters.heads;
        branch.norm = loader.create_tensor(root + ".norm.weight", {w});
        if (is_audio) {
            patch_embedding = create_linear(loader, root + ".embeddings.patch_embedding", h.patch * h.patch, w, false);
            register_tokens = loader.create_tensor(root + ".embeddings.register_tokens", {w, 1, 1});
            patch_norm = loader.create_tensor(root + ".embeddings.layernorm.weight", {w});
        } else
            text_embeddings = loader.create_tensor(root + ".embed_tokens.weight", {w, h.vocab});
        for (int i = 0; i < parameters.layers; ++i) {
            const auto p = root + ".encoder.layers." + std::to_string(i);
            Layer layer;
            layer.input_norm = loader.create_tensor(p + ".input_layernorm.weight", {w});
            layer.post_attention_norm = loader.create_tensor(p + ".post_attention_layernorm.weight", {w});
            layer.qkv = create_linear(loader, p + ".self_attn.qkv_proj", w, 3 * w, is_audio);
            layer.attention_output = create_linear(loader, p + ".self_attn.o_proj", w, w, is_audio);
            layer.gate_up = create_linear(loader, p + ".mlp.gate_up_proj", w, 2 * m, is_audio);
            layer.down = create_linear(loader, p + ".mlp.down_proj", m, w, is_audio);
            if (is_audio) {
                layer.attention_scale = loader.create_tensor(p + ".layer_scale1.lambda1", {w});
                layer.mlp_scale = loader.create_tensor(p + ".layer_scale2.lambda1", {w});
            } else {
                layer.q_norm = loader.create_tensor(p + ".self_attn.q_norm.weight", {d});
                layer.k_norm = loader.create_tensor(p + ".self_attn.k_norm.weight", {d});
            }
            branch.layers.push_back(layer);
        }
        const auto p = root + ".head";
        auto &head = branch.head;
        head.probe = loader.create_tensor(p + ".probe", {w, 1, 1});
        head.query = create_linear(loader, p + ".q_proj", w, w, is_audio);
        head.key_value = create_linear(loader, p + ".kv_proj", w, 2 * w, is_audio);
        if (is_audio) {
            head.output = create_linear(loader, p + ".o_proj", w, w);
            head.norm = loader.create_tensor(p + ".layernorm.weight", {w});
            head.fc1 = create_linear(loader, p + ".fc1", w, m);
            head.fc2 = create_linear(loader, p + ".fc2", m, w);
        } else {
            head.q_norm = loader.create_tensor(p + ".q_norm.weight", {d});
            head.k_norm = loader.create_tensor(p + ".k_norm.weight", {d});
        }
    }
    const int projection = h.projection, aw = audio_branch.h.hidden, tw = text_branch.h.hidden;
    audio_projection_input = create_linear(loader, "audio_projection.linear1", aw, projection);
    audio_projection_output = create_linear(loader, "audio_projection.linear2", projection, projection);
    text_projection = create_linear(loader, "text_projection", tw, projection, false);
    // Required checkpoint metadata tensor; inference does not consume its value.
    (void)loader.create_tensor("logit_scale", {1});
}
TextParameters Clap2Model::text_parameters() const { return {h.bos, h.eos, static_cast<size_t>(h.positions)}; }
int Clap2Model::projection_dimension() const { return h.projection; }
int Clap2Model::mel_bins() const { return h.bins; }
int Clap2Model::sample_rate() const { return h.rate; }

ggml_tensor *Clap2Model::rms(Graph &g, ggml_tensor *x, ggml_tensor *weight, float epsilon) const {
    return ggml_mul(g.ctx(), ggml_rms_norm(g.ctx(), x, epsilon), g.weight(weight));
}
// Interleaved (even, odd) rotary layout. Audio allocates half the pairs to
// frequency and half to time; register tokens are deliberately not rotated.
ggml_tensor *Clap2Model::rotate(Graph &g, ggml_tensor *x, int heads, float theta, int time, int prefix) const {
    auto *c = g.ctx();
    const int w = x->ne[0], n = x->ne[1], d = w / heads, pairs = d / 2;
    std::vector<float> cosine(size_t(n) * pairs), sine(cosine.size());
    for (int t = 0; t < n; ++t)
        for (int p = 0; p < pairs; ++p) {
            int pos = t;
            int band = p, bands = pairs;
            if (time) {
                int index = std::max(0, t - prefix);
                pos = p < pairs / 2 ? index / time : index % time;
                band = p % (pairs / 2);
                bands = pairs / 2;
            }
            const float angle = t < prefix ? 0.f : float(pos) * std::pow(theta, -float(band) / bands);
            cosine[size_t(t) * pairs + p] = std::cos(angle);
            sine[size_t(t) * pairs + p] = std::sin(angle);
        }
    auto *cs = ggml_reshape_4d(c, g.input(cosine, pairs, n), 1, pairs, 1, n);
    auto *sn = ggml_reshape_4d(c, g.input(sine, pairs, n), 1, pairs, 1, n);
    x = ggml_reshape_4d(c, x, 2, pairs, heads, n);
    auto *even = ggml_view_4d(c, x, 1, pairs, heads, n, x->nb[1], x->nb[2], x->nb[3], 0);
    auto *odd = ggml_view_4d(c, x, 1, pairs, heads, n, x->nb[1], x->nb[2], x->nb[3], sizeof(float));
    auto *a = ggml_sub(c, ggml_mul(c, even, cs), ggml_mul(c, odd, sn));
    auto *b = ggml_add(c, ggml_mul(c, even, sn), ggml_mul(c, odd, cs));
    return ggml_reshape_2d(c, ggml_concat(c, a, b, 0), w, n);
}
ggml_tensor *Clap2Model::attend(Graph &g, ggml_tensor *q, ggml_tensor *k, ggml_tensor *v, int heads,
                                ggml_tensor *mask) const {
    auto *c = g.ctx();
    const int w = q->ne[0], d = w / heads, nq = q->ne[1], nk = k->ne[1];
    q = ggml_permute(c, ggml_reshape_3d(c, q, d, heads, nq), 0, 2, 1, 3);
    k = ggml_permute(c, ggml_reshape_3d(c, k, d, heads, nk), 0, 2, 1, 3);
    v = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, d, heads, nk), 1, 2, 0, 3));
    auto *scores = ggml_scale(c, g.matmul(k, q), 1.f / std::sqrt(float(d)));
    if (mask)
        scores = ggml_add(c, scores, mask);
    auto *out = g.matmul(v, ggml_soft_max(c, scores));
    return ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, out, 0, 2, 1, 3)), w, nq);
}
ggml_tensor *Clap2Model::encode(Graph &g, ggml_tensor *x, bool is_audio, int time, ggml_tensor *mask) const {
    auto *c = g.ctx();
    const auto &branch = is_audio ? audio_branch : text_branch;
    const int w = branch.h.hidden, heads = branch.h.heads, d = w / heads;
    const float eps = branch.h.epsilon, theta = branch.h.theta;
    auto qknorm = [&](ggml_tensor *z, ggml_tensor *weight) {
        return ggml_reshape_2d(c, rms(g, ggml_reshape_3d(c, z, d, heads, z->ne[1]), weight, eps), w, z->ne[1]);
    };
    for (const auto &layer : branch.layers) {
        auto *qkv = g.linear(rms(g, x, layer.input_norm, eps), layer.qkv);
        auto *q = g.slice(qkv, 0, w), *k = g.slice(qkv, w, w), *v = g.slice(qkv, 2 * w, w);
        if (!is_audio) {
            q = qknorm(q, layer.q_norm);
            k = qknorm(k, layer.k_norm);
        }
        q = rotate(g, q, heads, theta, time, is_audio ? 1 : 0);
        k = rotate(g, k, heads, theta, time, is_audio ? 1 : 0);
        auto *y = g.linear(attend(g, q, k, v, heads, mask), layer.attention_output);
        if (is_audio)
            y = ggml_mul(c, y, g.weight(layer.attention_scale));
        x = ggml_add(c, x, y);
        auto *gu = g.linear(rms(g, x, layer.post_attention_norm, eps), layer.gate_up);
        const int m = branch.h.intermediate;
        // HF gate_up_proj is gate-first in both towers. ModernText.w12 is
        // value-first upstream, but the pinned HF export swaps its two halves.
        y = ggml_mul(c, ggml_silu(c, g.slice(gu, 0, m)), g.slice(gu, m, m));
        y = g.linear(y, layer.down);
        if (is_audio)
            y = ggml_mul(c, y, g.weight(layer.mlp_scale));
        x = ggml_add(c, x, y);
    }
    return rms(g, x, branch.norm, eps);
}
ggml_tensor *Clap2Model::pool(Graph &g, ggml_tensor *x, bool is_audio, ggml_tensor *mask) const {
    auto *c = g.ctx();
    const auto &branch = is_audio ? audio_branch : text_branch;
    const auto &head = branch.head;
    const int w = branch.h.hidden, heads = branch.h.heads, d = w / heads;
    const float eps = branch.h.epsilon;
    if (is_audio) {
        // NaFlex MAP excludes the register prefix (pool_include_prefix=False).
        x = ggml_cont(c, ggml_view_2d(c, x, x->ne[0], x->ne[1] - 1, x->nb[1], x->nb[1]));
    }
    auto *probe = ggml_reshape_2d(c, g.weight(head.probe), w, 1);
    auto *q = g.linear(probe, head.query);
    auto *kv = g.linear(x, head.key_value);
    auto *k = g.slice(kv, 0, w), *v = g.slice(kv, w, w);
    if (!is_audio) {
        q = ggml_reshape_2d(c, rms(g, ggml_reshape_3d(c, q, d, heads, 1), head.q_norm, eps), w, 1);
        k = ggml_reshape_2d(c, rms(g, ggml_reshape_3d(c, k, d, heads, x->ne[1]), head.k_norm, eps), w, x->ne[1]);
    }
    auto *out = attend(g, q, k, v, heads, mask);
    if (is_audio) {
        out = g.linear(out, head.output);
        auto *y = g.linear(rms(g, out, head.norm, eps), head.fc1);
        y = ggml_gelu_erf(c, y);
        out = ggml_add(c, out, g.linear(y, head.fc2));
        out = g.linear(out, audio_projection_input);
        out = g.linear(ggml_gelu_erf(c, out), audio_projection_output);
    } else
        out = g.linear(out, text_projection);
    return g.l2_normalize(out);
}
std::vector<float> Clap2Model::text(Context &context, const std::vector<int> &ids) const {
    const auto &runtime = context.runtime(this, loader, true);
    if (ids.empty() || ids.size() > size_t(h.positions))
        throw std::runtime_error("CLAP: invalid CLAP2 text length");
    std::vector<float> mask(ids.size());
    bool valid = false;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= h.vocab)
            throw std::runtime_error("CLAP: CLAP2 token out of range");
        mask[i] = ids[i] == h.pad ? -INFINITY : 0.f;
        valid |= mask[i] == 0;
    }
    if (!valid)
        throw std::runtime_error("CLAP: CLAP2 text is entirely padding");
    auto result = runtime.run_cached("clap2.text", {GraphInput(ids), GraphInput(mask, ids.size(), 1)}, [&](Graph &g) {
        auto *x = g.embedding(text_embeddings, g.dynamic_input(0));
        auto *m = g.dynamic_input(1);
        return std::vector<ggml_tensor *>{pool(g, encode(g, x, false, 0, m), false, m)};
    });
    return result.front().values;
}
HostMatrix Clap2Model::frontend(const std::vector<float> &pcm) const {
    const int bins = h.bins, rate = h.rate, nfft = h.nfft, hop = h.hop;
    const float fmin = h.fmin, fmax = h.fmax;
    if (pcm.empty() || pcm.size() > size_t(INT32_MAX) ||
        !std::all_of(pcm.begin(), pcm.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("CLAP: invalid CLAP2 waveform");
    std::vector<float> padded = pcm;
    padded.resize(std::max(pcm.size(), size_t(nfft)), 0.f);
    const int samples = padded.size(), frames = samples / hop + 1, freqs = nfft / 2 + 1;
    HostMatrix mel(frames, bins);
    std::vector<float> window(nfft), edges(bins + 2), filters(size_t(freqs) * bins), power(freqs);
    for (int i = 0; i < nfft; ++i)
        window[i] = 0.5f * (1.f - std::cos(2.f * static_cast<float>(std::numbers::pi) * float(i) / nfft));
    const float lo = float(2595.0 * std::log10(1.0 + fmin / 700.0));
    const float hi = float(2595.0 * std::log10(1.0 + fmax / 700.0));
    for (int i = 0; i < bins + 2; ++i) {
        const float m = lo + (hi - lo) * float(i) / float(bins + 1);
        edges[i] = 700.f * (std::pow(10.f, m / 2595.f) - 1.f);
    }
    for (int f = 0; f < freqs; ++f)
        for (int b = 0; b < bins; ++b) {
            float hz = float(rate / 2) * float(f) / float(freqs - 1);
            filters[size_t(f) * bins + b] =
                std::max(0.f, std::min((hz - edges[b]) / (edges[b + 1] - edges[b]),
                                       (edges[b + 2] - hz) / (edges[b + 2] - edges[b + 1])));
        }
    std::vector<std::complex<double>> spectrum(nfft);
    for (int t = 0; t < frames; ++t) {
        for (int i = 0; i < nfft; ++i) {
            int index = t * hop + i - nfft / 2;
            if (index < 0)
                index = -index;
            if (index >= samples)
                index = 2 * samples - 2 - index;
            spectrum[i] = padded[index] * window[i];
        }
        clap::fft(spectrum);
        for (int f = 0; f < freqs; ++f) {
            const float re = float(spectrum[f].real()), im = float(spectrum[f].imag());
            const float magnitude = std::sqrt(re * re + im * im);
            power[f] = magnitude * magnitude;
        }
        for (int b = 0; b < bins; ++b) {
            float sum = 0;
            for (int f = 0; f < freqs; ++f)
                sum += power[f] * filters[size_t(f) * bins + b];
            mel[t][b] = 10.f * std::log10(std::max(sum, 1e-10f));
        }
    }
    return mel;
}
HostMatrix Clap2Model::patches(const HostMatrix &mel) const {
    const int patch = h.patch, bins = h.bins;
    const int time = (mel.rows + patch - 1) / patch, freq = bins / patch;
    HostMatrix out(time * freq, patch * patch);
    const float floor = *std::min_element(mel.data.begin(), mel.data.end());
    for (int f = 0; f < freq; ++f)
        for (int t = 0; t < time; ++t)
            for (int pf = 0; pf < patch; ++pf)
                for (int pt = 0; pt < patch; ++pt) {
                    int frame = t * patch + pt;
                    out[f * time + t][pf * patch + pt] = frame < mel.rows ? mel[frame][f * patch + pf] : floor;
                }
    return out;
}
std::vector<float> Clap2Model::waveform_embedding(Context &context, const std::vector<float> &pcm,
                                                  size_t offset) const {
    const auto &runtime = context.runtime(this, loader, true);
    if (offset)
        throw std::runtime_error("CLAP: CLAP2 processes the complete waveform; crop offsets are unsupported");
    const auto features = frontend(pcm);
    const auto input = patches(features);
    const int time = (features.rows + h.patch - 1) / h.patch;
    auto result = runtime.run_cached("clap2.audio", {GraphInput(input)}, [&](Graph &g) {
        auto *x = g.linear(g.dynamic_input(0), patch_embedding);
        auto *reg = ggml_reshape_2d(g.ctx(), g.weight(register_tokens), x->ne[0], 1);
        x = ggml_concat(g.ctx(), reg, x, 1);
        x = rms(g, x, patch_norm, audio_branch.h.epsilon);
        x = encode(g, x, true, time, nullptr);
        return std::vector<ggml_tensor *>{pool(g, x, true, nullptr)};
    });
    return result.front().values;
}
AudioFrames Clap2Model::audio_frames(Context &, const std::vector<float> &) const {
    throw std::runtime_error("CLAP: CLAP2 exposes pooled audio embeddings, not frame representations");
}
} // namespace clap::detail
