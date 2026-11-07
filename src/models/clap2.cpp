#include "clap2.h"
#include "audio.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace clap::detail {
Clap2Model::Clap2Model(const ModelLoader &source, int threads) : loader(source), runtime(source, threads, true) {
    load_arch_hparams();
    load_arch_tensors();
}
void Clap2Model::load_arch_hparams() {
    config = nlohmann::json::parse(loader.metadata("clap2.config"));
    processor = nlohmann::json::parse(loader.metadata("clap2.processor_config"));
    audio = config.at("audio_config");
    txt = config.at("text_config");
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
}
void Clap2Model::load_arch_tensors() const {
    for (bool is_audio : {true, false}) {
        const auto &j = is_audio ? audio : txt;
        const std::string root = is_audio ? "audio_model" : "text_model";
        const int w = j.at("hidden_size"), m = j.at("intermediate_size"),
                  d = w / j.at("num_attention_heads").get<int>();
        auto norm = [&](const std::string &s, int size) { loader.shape(s + ".weight", {size}); };
        norm(root + ".norm", w);
        if (is_audio) {
            loader.linear_shape(root + ".embeddings.patch_embedding", patch * patch, w, false);
            loader.shape(root + ".embeddings.register_tokens", {w, 1, 1});
            norm(root + ".embeddings.layernorm", w);
        } else
            loader.shape(root + ".embed_tokens.weight", {w, txt.at("vocab_size").get<int>()});
        for (int i = 0; i < j.at("num_hidden_layers").get<int>(); ++i) {
            const auto p = root + ".encoder.layers." + std::to_string(i);
            norm(p + ".input_layernorm", w);
            norm(p + ".post_attention_layernorm", w);
            loader.linear_shape(p + ".self_attn.qkv_proj", w, 3 * w, is_audio);
            loader.linear_shape(p + ".self_attn.o_proj", w, w, is_audio);
            loader.linear_shape(p + ".mlp.gate_up_proj", w, 2 * m, is_audio);
            loader.linear_shape(p + ".mlp.down_proj", m, w, is_audio);
            if (is_audio) {
                loader.shape(p + ".layer_scale1.lambda1", {w});
                loader.shape(p + ".layer_scale2.lambda1", {w});
            } else {
                norm(p + ".self_attn.q_norm", d);
                norm(p + ".self_attn.k_norm", d);
            }
        }
        const auto h = root + ".head";
        loader.shape(h + ".probe", {w, 1, 1});
        loader.linear_shape(h + ".q_proj", w, w, is_audio);
        loader.linear_shape(h + ".kv_proj", w, 2 * w, is_audio);
        if (is_audio) {
            loader.linear_shape(h + ".o_proj", w, w);
            norm(h + ".layernorm", w);
            loader.linear_shape(h + ".fc1", w, m);
            loader.linear_shape(h + ".fc2", m, w);
        } else {
            norm(h + ".q_norm", d);
            norm(h + ".k_norm", d);
        }
    }
    const int projection = config.at("projection_dim"), aw = audio.at("hidden_size"), tw = txt.at("hidden_size");
    loader.linear_shape("audio_projection.linear1", aw, projection);
    loader.linear_shape("audio_projection.linear2", projection, projection);
    loader.linear_shape("text_projection", tw, projection, false);
    loader.shape("logit_scale", {1});
}
TextParameters Clap2Model::text_parameters() const {
    return {txt.at("bos_token_id"), txt.at("eos_token_id"), txt.at("max_position_embeddings")};
}
int Clap2Model::projection_dimension() const { return config.at("projection_dim"); }
int Clap2Model::mel_bins() const { return bins; }
int Clap2Model::sample_rate() const { return rate; }

ggml_tensor *Clap2Model::rms(Graph &g, ggml_tensor *x, const std::string &p, float epsilon) const {
    return ggml_mul(g.ctx(), ggml_rms_norm(g.ctx(), x, epsilon), g.weight(p + ".weight"));
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
    const auto &j = is_audio ? audio : txt;
    const std::string root = is_audio ? "audio_model" : "text_model";
    const int w = j.at("hidden_size"), heads = j.at("num_attention_heads"), d = w / heads;
    const float eps = j.at("rms_norm_eps"), theta = j.at("rope_parameters").at("rope_theta");
    auto qknorm = [&](ggml_tensor *z, const std::string &p) {
        return ggml_reshape_2d(c, rms(g, ggml_reshape_3d(c, z, d, heads, z->ne[1]), p, eps), w, z->ne[1]);
    };
    for (int i = 0; i < j.at("num_hidden_layers").get<int>(); ++i) {
        const auto p = root + ".encoder.layers." + std::to_string(i);
        auto *qkv = g.linear(rms(g, x, p + ".input_layernorm", eps), p + ".self_attn.qkv_proj", is_audio);
        auto *q = g.slice(qkv, 0, w), *k = g.slice(qkv, w, w), *v = g.slice(qkv, 2 * w, w);
        if (!is_audio) {
            q = qknorm(q, p + ".self_attn.q_norm");
            k = qknorm(k, p + ".self_attn.k_norm");
        }
        q = rotate(g, q, heads, theta, time, is_audio ? 1 : 0);
        k = rotate(g, k, heads, theta, time, is_audio ? 1 : 0);
        auto *y = g.linear(attend(g, q, k, v, heads, mask), p + ".self_attn.o_proj", is_audio);
        if (is_audio)
            y = ggml_mul(c, y, g.weight(p + ".layer_scale1.lambda1"));
        x = ggml_add(c, x, y);
        auto *gu = g.linear(rms(g, x, p + ".post_attention_layernorm", eps), p + ".mlp.gate_up_proj", is_audio);
        const int m = j.at("intermediate_size");
        // HF gate_up_proj is gate-first in both towers. ModernText.w12 is
        // value-first upstream, but the pinned HF export swaps its two halves.
        y = ggml_mul(c, ggml_silu(c, g.slice(gu, 0, m)), g.slice(gu, m, m));
        y = g.linear(y, p + ".mlp.down_proj", is_audio);
        if (is_audio)
            y = ggml_mul(c, y, g.weight(p + ".layer_scale2.lambda1"));
        x = ggml_add(c, x, y);
    }
    return rms(g, x, root + ".norm", eps);
}
ggml_tensor *Clap2Model::pool(Graph &g, ggml_tensor *x, bool is_audio, ggml_tensor *mask) const {
    auto *c = g.ctx();
    const auto &j = is_audio ? audio : txt;
    const std::string p = is_audio ? "audio_model.head" : "text_model.head";
    const int w = j.at("hidden_size"), heads = j.at("num_attention_heads"), d = w / heads;
    const float eps = j.at("rms_norm_eps");
    if (is_audio) {
        // NaFlex MAP excludes the register prefix (pool_include_prefix=False).
        x = ggml_cont(c, ggml_view_2d(c, x, x->ne[0], x->ne[1] - 1, x->nb[1], x->nb[1]));
    }
    auto *probe = ggml_reshape_2d(c, g.weight(p + ".probe"), w, 1);
    auto *q = g.linear(probe, p + ".q_proj", is_audio);
    auto *kv = g.linear(x, p + ".kv_proj", is_audio);
    auto *k = g.slice(kv, 0, w), *v = g.slice(kv, w, w);
    if (!is_audio) {
        q = ggml_reshape_2d(c, rms(g, ggml_reshape_3d(c, q, d, heads, 1), p + ".q_norm", eps), w, 1);
        k = ggml_reshape_2d(c, rms(g, ggml_reshape_3d(c, k, d, heads, x->ne[1]), p + ".k_norm", eps), w, x->ne[1]);
    }
    auto *out = attend(g, q, k, v, heads, mask);
    if (is_audio) {
        out = g.linear(out, p + ".o_proj");
        auto *y = g.linear(rms(g, out, p + ".layernorm", eps), p + ".fc1");
        y = ggml_gelu_erf(c, y);
        out = ggml_add(c, out, g.linear(y, p + ".fc2"));
        out = g.linear(out, "audio_projection.linear1");
        out = g.linear(ggml_gelu_erf(c, out), "audio_projection.linear2");
    } else
        out = g.linear(out, "text_projection", false);
    return g.l2_normalize(out);
}
std::vector<float> Clap2Model::text(const std::vector<int> &ids) const {
    if (ids.empty() || ids.size() > size_t(txt.at("max_position_embeddings").get<int>()))
        throw std::runtime_error("CLAP: invalid CLAP2 text length");
    std::vector<float> mask(ids.size());
    bool valid = false;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= txt.at("vocab_size").get<int>())
            throw std::runtime_error("CLAP: CLAP2 token out of range");
        mask[i] = ids[i] == txt.at("pad_token_id").get<int>() ? -INFINITY : 0.f;
        valid |= mask[i] == 0;
    }
    if (!valid)
        throw std::runtime_error("CLAP: CLAP2 text is entirely padding");
    auto result = runtime.run_cached("clap2.text", {GraphInput(ids), GraphInput(mask, ids.size(), 1)}, [&](Graph &g) {
        auto *x = g.embedding(loader.weight("text_model.embed_tokens.weight"), g.dynamic_input(0));
        auto *m = g.dynamic_input(1);
        return std::vector<ggml_tensor *>{pool(g, encode(g, x, false, 0, m), false, m)};
    });
    return result.front().values;
}
HostMatrix Clap2Model::frontend(const std::vector<float> &pcm) const {
    if (pcm.empty() || pcm.size() > size_t(INT32_MAX) ||
        !std::all_of(pcm.begin(), pcm.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("CLAP: invalid CLAP2 waveform");
    std::vector<float> padded = pcm;
    padded.resize(std::max(pcm.size(), size_t(nfft)), 0.f);
    const int samples = padded.size(), frames = samples / hop + 1, freqs = nfft / 2 + 1;
    HostMatrix mel(frames, bins);
    std::vector<float> window(nfft), edges(bins + 2), filters(size_t(freqs) * bins), power(freqs);
    const float pi = float(std::acos(-1.0));
    for (int i = 0; i < nfft; ++i)
        window[i] = 0.5f * (1.f - std::cos(2.f * pi * float(i) / nfft));
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
std::vector<float> Clap2Model::waveform_embedding(const std::vector<float> &pcm, size_t offset) const {
    if (offset)
        throw std::runtime_error("CLAP: CLAP2 processes the complete waveform; crop offsets are unsupported");
    const auto features = frontend(pcm);
    const auto input = patches(features);
    const int time = (features.rows + patch - 1) / patch;
    auto result = runtime.run_cached("clap2.audio", {GraphInput(input)}, [&](Graph &g) {
        auto *x = g.linear(g.dynamic_input(0), "audio_model.embeddings.patch_embedding", false);
        auto *reg = ggml_reshape_2d(g.ctx(), g.weight("audio_model.embeddings.register_tokens"), x->ne[0], 1);
        x = ggml_concat(g.ctx(), reg, x, 1);
        x = rms(g, x, "audio_model.embeddings.layernorm", audio.at("rms_norm_eps"));
        x = encode(g, x, true, time, nullptr);
        return std::vector<ggml_tensor *>{pool(g, x, true, nullptr)};
    });
    return result.front().values;
}
AudioFrames Clap2Model::audio_frames(const std::vector<float> &) const {
    throw std::runtime_error("CLAP: CLAP2 exposes pooled audio embeddings, not frame representations");
}
} // namespace clap::detail
