#include "clap_model.h"

#include <ggml.h>
#include <ggml-cpu.h>
#include <gguf.h>
#include <nlohmann/json.hpp>
#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace clap {
namespace {
using json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;
void check(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error("CLAP: " + message);
}
struct Matrix {
    int rows, cols;
    std::vector<float> data;
    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0) {
        check(r > 0 && c > 0, "invalid matrix shape");
    }
    float *operator[](int r) { return data.data() + static_cast<size_t>(r) * cols; }
    const float *operator[](int r) const { return data.data() + static_cast<size_t>(r) * cols; }
};
void add(Matrix &a, const Matrix &b) {
    check(a.rows == b.rows && a.cols == b.cols, "residual shape mismatch");
    for (size_t i = 0; i < a.data.size(); ++i) a.data[i] += b.data[i];
}
void gelu(Matrix &x) {
    for (float &v : x.data) v = 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f)));
}
float scalar(const ggml_tensor *t, size_t i) {
    if (t->type == GGML_TYPE_F32) return static_cast<const float *>(t->data)[i];
    return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(t->data)[i]);
}
std::string utf8(int cp) {
    icu::UnicodeString u; u.append(static_cast<UChar32>(cp));
    std::string s; u.toUTF8String(s); return s;
}
// PyTorch bicubic interpolation, A=-0.75, align_corners=True.
float cubic(float x) {
    x = std::abs(x);
    if (x <= 1) return ((1.25f * x - 2.25f) * x) * x + 1;
    if (x < 2) return ((-0.75f * x + 3.75f) * x - 6) * x + 3;
    return 0;
}
void fft(std::vector<std::complex<double>> &a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const auto root = std::polar(1.0, -2 * pi / static_cast<double>(len));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1);
            for (size_t j = 0; j < len / 2; ++j) {
                const auto u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v; a[i + j + len / 2] = u - v; w *= root;
            }
        }
    }
}
double hz_to_mel(double hz) {
    return hz < 1000 ? hz / (200.0 / 3) : 15 + std::log(hz / 1000) / (std::log(6.4) / 27);
}
double mel_to_hz(double mel) {
    return mel < 15 ? mel * (200.0 / 3) : 1000 * std::exp((mel - 15) * (std::log(6.4) / 27));
}
} // namespace

std::vector<float> log_mel(const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
    check(sample_rate == 48000, "audio must be mono 48000 Hz (resample before inference)");
    check(!pcm.empty(), "empty waveform");
    check(std::all_of(pcm.begin(), pcm.end(), [](float x) { return std::isfinite(x); }), "non-finite waveform");
    constexpr size_t length = 480000;
    std::vector<double> wav(length, 0);
    if (pcm.size() > length) {
        check(crop_offset <= pcm.size() - length, "crop offset out of range");
        std::copy_n(pcm.data() + crop_offset, length, wav.data());
    } else {
        check(crop_offset == 0, "crop offset requires audio longer than 10 seconds");
        const size_t repeats = length / pcm.size();
        for (size_t i = 0; i < repeats; ++i) std::copy(pcm.begin(), pcm.end(), wav.begin() + i * pcm.size());
    }
    double filters[64][513];
    double boundaries[66];
    const double low = hz_to_mel(50), high = hz_to_mel(14000);
    for (int i = 0; i < 66; ++i) boundaries[i] = mel_to_hz(low + (high - low) * i / 65);
    for (int m = 0; m < 64; ++m) for (int f = 0; f <= 512; ++f) {
        const double hz = 48000.0 * f / 1024;
        filters[m][f] = std::max(0.0, std::min((hz - boundaries[m]) / (boundaries[m + 1] - boundaries[m]),
                                                   (boundaries[m + 2] - hz) / (boundaries[m + 2] - boundaries[m + 1])))
                         * (2 / (boundaries[m + 2] - boundaries[m]));
    }
    std::vector<float> output(1001 * 64);
    std::vector<std::complex<double>> frame(1024);
    double power[513], window[1024];
    for (int i = 0; i < 1024; ++i) window[i] = 0.5 - 0.5 * std::cos(2 * pi * i / 1024);
    for (int t = 0; t <= 1000; ++t) {
        for (int i = 0; i < 1024; ++i) {
            int pos = t * 480 + i - 512;
            if (pos < 0) pos = -pos;
            if (pos >= static_cast<int>(length)) pos = 2 * static_cast<int>(length) - 2 - pos;
            frame[i] = wav[pos] * window[i];
        }
        fft(frame);
        for (int i = 0; i <= 512; ++i) power[i] = std::norm(frame[i]);
        for (int m = 0; m < 64; ++m) {
            double sum = 0;
            for (int f = 0; f <= 512; ++f) sum += filters[m][f] * power[f];
            output[t * 64 + m] = static_cast<float>(10 * std::log10(std::max(sum, 1e-10)));
        }
    }
    return output;
}

struct Model::Impl {
    gguf_context *gguf = nullptr;
    ggml_context *weights = nullptr;
    void *mapping = nullptr;
    size_t mapping_size = 0;
#ifdef _WIN32
    std::vector<char> storage;
#endif
    json config, a, t, tokenizer;
    int threads = 1;
    std::unordered_map<std::string, int> vocab;
    std::map<std::pair<std::string, std::string>, int> ranks;
    std::vector<json> special;
    std::string byte_chars[256];
    std::unique_ptr<icu::RegexPattern> pattern;

    ~Impl() {
        if (weights) ggml_free(weights);
        if (gguf) gguf_free(gguf);
#ifndef _WIN32
        if (mapping && mapping != MAP_FAILED) munmap(mapping, mapping_size);
#endif
    }
    ggml_tensor *weight(const std::string &name) const {
        auto *w = ggml_get_tensor(weights, name.c_str());
        check(w != nullptr, "missing tensor " + name);
        return w;
    }
    void shape(const std::string &name, std::initializer_list<int64_t> dims) const {
        const auto *w = weight(name);
        int i = 0;
        for (auto d : dims) { check(w->ne[i++] == d, "wrong tensor shape: " + name); }
        while (i < GGML_MAX_DIMS) check(w->ne[i++] == 1, "wrong tensor rank: " + name);
    }
    void linear_shape(const std::string &name, int in, int out, bool bias = true) const {
        shape(name + ".weight", {in, out});
        if (bias) shape(name + ".bias", {out});
    }
    void norm_shape(const std::string &name, int dim) const {
        shape(name + ".weight", {dim}); shape(name + ".bias", {dim});
    }
    void validate() const {
        check(config.at("model_type") == "clap", "not a CLAP model");
        check(!a.at("enable_fusion").get<bool>(), "fusion models are not supported");
        check(a.at("depths").size() == 4 && a.at("num_attention_heads").size() == 4, "expected four HTSAT stages");
        check(a.at("patch_size") == 4 && a.at("patch_stride") == json::array({4, 4}) &&
              a.at("patch_embed_input_channels") == 1 && a.at("qkv_bias").get<bool>() &&
              a.at("enable_patch_layer_norm").get<bool>() && a.at("flatten_patch_embeds").get<bool>(), "unsupported patch embedding");
        check(a.at("hidden_act") == "gelu" && t.at("hidden_act") == "gelu", "only exact GELU is supported");
        check(a.at("projection_hidden_act") == "relu" && t.at("projection_hidden_act") == "relu", "only ReLU projections supported");
        check(t.value("position_embedding_type", std::string("absolute")) == "absolute", "unsupported text position embedding type");
        const int spec = a.at("spec_size"), bins = a.at("num_mel_bins"), window = a.at("window_size");
        check(spec > 0 && spec % 32 == 0 && bins > 0 && spec % bins == 0 && window > 0 &&
              spec / 32 >= spec / bins, "invalid spectrogram geometry");
        int dim = a.at("patch_embeds_hidden_size");
        check(dim > 0 && dim * 8 == a.at("hidden_size"), "audio hidden size mismatch");
        shape("audio.patch_embed.proj.weight", {4, 4, 1, dim});
        shape("audio.patch_embed.proj.bias", {dim});
        norm_shape("audio.patch_embed.norm", dim);
        norm_shape("audio.batch_norm", bins);
        shape("audio.batch_norm.running_mean", {bins}); shape("audio.batch_norm.running_var", {bins});
        for (int s = 0; s < 4; ++s) {
            const int heads = a.at("num_attention_heads")[s], depth = a.at("depths")[s];
            check(heads > 0 && dim % heads == 0 && depth > 0, "invalid audio attention dimensions");
            check(spec / (4 << s) >= window, "window larger than stage resolution is unsupported");
            for (int b = 0; b < depth; ++b) {
                const std::string p = "audio.layers." + std::to_string(s) + ".blocks." + std::to_string(b);
                norm_shape(p + ".layernorm_before", dim); norm_shape(p + ".layernorm_after", dim);
                for (const auto *q : {"query", "key", "value"}) linear_shape(p + ".attention.self." + q, dim, dim);
                shape(p + ".attention.self.rel_bias", {heads, (2 * window - 1) * (2 * window - 1)});
                linear_shape(p + ".attention.output.dense", dim, dim);
                const int inner = static_cast<int>(dim * a.at("mlp_ratio").get<double>());
                linear_shape(p + ".intermediate.dense", dim, inner); linear_shape(p + ".output.dense", inner, dim);
            }
            if (s < 3) {
                const auto p = "audio.layers." + std::to_string(s) + ".downsample";
                norm_shape(p + ".norm", dim * 4); linear_shape(p + ".reduction", dim * 4, dim * 2, false);
            }
            dim *= 2;
        }
        norm_shape("audio.norm", dim / 2);
        dim = t.at("hidden_size");
        const int heads = t.at("num_attention_heads"), layers = t.at("num_hidden_layers");
        check(dim > 0 && heads > 0 && dim % heads == 0 && layers > 0 &&
              t.at("type_vocab_size").get<int>() > 0, "invalid text dimensions");
        shape("text.embeddings.word_embeddings.weight", {dim, t.at("vocab_size").get<int>()});
        shape("text.embeddings.position_embeddings.weight", {dim, t.at("max_position_embeddings").get<int>()});
        shape("text.embeddings.token_type_embeddings.weight", {dim, t.at("type_vocab_size").get<int>()});
        norm_shape("text.embeddings.LayerNorm", dim);
        for (int l = 0; l < layers; ++l) {
            const auto p = "text.encoder.layer." + std::to_string(l);
            for (const auto *q : {"query", "key", "value"}) linear_shape(p + ".attention.self." + q, dim, dim);
            linear_shape(p + ".attention.output.dense", dim, dim); norm_shape(p + ".attention.output.LayerNorm", dim);
            const int inner = t.at("intermediate_size");
            linear_shape(p + ".intermediate.dense", dim, inner); linear_shape(p + ".output.dense", inner, dim);
            norm_shape(p + ".output.LayerNorm", dim);
        }
        linear_shape("text.pooler.dense", dim, dim);
        for (const auto *branch : {"audio", "text"}) {
            const auto &c = std::string(branch) == "audio" ? a : t;
            const int pd = c.at("projection_dim");
            check(pd > 0 && pd == config.at("projection_dim"), "projection dimensions disagree");
            linear_shape(std::string(branch) + "_proj.linear1", c.at("hidden_size"), pd);
            linear_shape(std::string(branch) + "_proj.linear2", pd, pd);
        }
    }
    void load(const std::string &path, int nthreads) {
        threads = nthreads > 0 ? nthreads : std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
        gguf = gguf_init_from_file(path.c_str(), {true, &weights});
        check(gguf && weights, "cannot load GGUF " + path);
        auto metadata = [&](const char *key) {
            const auto id = gguf_find_key(gguf, key);
            check(id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_STRING, std::string("missing string metadata: ") + key);
            return std::string(gguf_get_val_str(gguf, id));
        };
        check(metadata("general.architecture") == "clap", "unsupported GGUF architecture");
        config = json::parse(metadata("clap.config")); a = config.at("audio_config"); t = config.at("text_config");
        tokenizer = json::parse(metadata("clap.tokenizer"));
#ifndef _WIN32
        const int fd = open(path.c_str(), O_RDONLY);
        check(fd >= 0, "cannot open weights");
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); check(false, "cannot stat weights"); }
        mapping_size = static_cast<size_t>(st.st_size);
        mapping = mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        check(mapping != MAP_FAILED, "cannot mmap weights");
#else
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        check(static_cast<bool>(stream), "cannot open weights");
        mapping_size = static_cast<size_t>(stream.tellg()); storage.resize(mapping_size);
        stream.seekg(0); stream.read(storage.data(), static_cast<std::streamsize>(storage.size()));
        check(static_cast<bool>(stream), "cannot read weights"); mapping = storage.data();
#endif
        const auto data_offset = gguf_get_data_offset(gguf);
        check(data_offset <= mapping_size, "truncated GGUF data");
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
            auto *w = weight(gguf_get_tensor_name(gguf, i));
            check(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16, "only F32/F16 weights supported");
            const size_t offset = gguf_get_tensor_offset(gguf, i);
            check(offset <= mapping_size - data_offset && ggml_nbytes(w) <= mapping_size - data_offset - offset, "truncated tensor data");
            w->data = static_cast<char *>(mapping) + data_offset + offset;
        }
        validate();
        check(tokenizer.at("model").at("type") == "BPE", "expected byte-level BPE");
        for (auto it = tokenizer["model"]["vocab"].begin(); it != tokenizer["model"]["vocab"].end(); ++it)
            vocab.emplace(it.key(), it.value().get<int>());
        int rank = 0;
        for (const auto &entry : tokenizer["model"]["merges"]) {
            std::string x, y;
            if (entry.is_array()) { x = entry.at(0); y = entry.at(1); }
            else {
                const std::string s = entry; const auto space = s.find(' ');
                check(space != std::string::npos, "invalid BPE merge"); x = s.substr(0, space); y = s.substr(space + 1);
            }
            ranks.emplace(std::make_pair(x, y), rank++);
        }
        if (tokenizer.contains("added_tokens")) for (const auto &v : tokenizer["added_tokens"]) {
            check(!v.at("content").get<std::string>().empty(), "empty added token");
            special.push_back(v);
        }
        int extra = 256;
        for (int i = 0; i < 256; ++i) {
            const bool regular = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174;
            byte_chars[i] = utf8(regular ? i : extra++);
        }
        UErrorCode status = U_ZERO_ERROR;
        pattern.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"), 0, status));
        check(U_SUCCESS(status), "cannot compile tokenizer Unicode regex");
    }
    Matrix linear(const Matrix &x, const std::string &name, bool bias = true) const {
        auto *w = weight(name + ".weight");
        check(w->ne[0] == x.cols && w->ne[2] == 1 && w->ne[3] == 1, "linear shape: " + name);
        Matrix result(x.rows, static_cast<int>(w->ne[1]));
        const size_t bytes = (x.data.size() + result.data.size()) * sizeof(float) + 2 * 1024 * 1024;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
        check(ctx != nullptr, "ggml allocation failed");
        auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, x.cols, x.rows);
        std::memcpy(input->data, x.data.data(), x.data.size() * sizeof(float));
        auto *output = ggml_mul_mat(ctx.get(), w, input);
        auto *graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        check(ggml_graph_compute_with_ctx(ctx.get(), graph, threads) == GGML_STATUS_SUCCESS, "ggml matrix multiply failed");
        std::memcpy(result.data.data(), output->data, result.data.size() * sizeof(float));
        if (bias) {
            const auto *b = weight(name + ".bias");
            for (int r = 0; r < result.rows; ++r) for (int c = 0; c < result.cols; ++c) result[r][c] += scalar(b, c);
        }
        return result;
    }
    Matrix norm(const Matrix &x, const std::string &name, float eps) const {
        const auto *w = weight(name + ".weight"), *b = weight(name + ".bias");
        Matrix y(x.rows, x.cols);
        for (int r = 0; r < x.rows; ++r) {
            double mean = 0, var = 0;
            for (int c = 0; c < x.cols; ++c) mean += x[r][c];
            mean /= x.cols;
            for (int c = 0; c < x.cols; ++c) { const double d = x[r][c] - mean; var += d * d; }
            const float scale = static_cast<float>(1 / std::sqrt(var / x.cols + eps));
            for (int c = 0; c < x.cols; ++c) y[r][c] = (x[r][c] - static_cast<float>(mean)) * scale * scalar(w, c) + scalar(b, c);
        }
        return y;
    }
    Matrix attention(const Matrix &x, const std::string &prefix, int heads, int tokens,
                     const std::vector<int> *mask = nullptr, int window = 0, int shift = 0, int grid = 0) const {
        const auto q = linear(x, prefix + ".query"), k = linear(x, prefix + ".key"), v = linear(x, prefix + ".value");
        Matrix result(x.rows, x.cols);
        const int d = x.cols / heads;
        const float scale = 1 / std::sqrt(static_cast<float>(d));
        const auto *bias = window ? weight(prefix + ".rel_bias") : nullptr;
        std::vector<float> scores(tokens);
        auto region = [&](int r, int c) {
            const auto section = [&](int p) { return p < grid - window ? 0 : p < grid - shift ? 1 : 2; };
            return section(r) * 3 + section(c);
        };
        for (int start = 0; start < x.rows; start += tokens) {
            const int win = start / tokens, windows_per_row = window ? grid / window : 0;
            for (int h = 0; h < heads; ++h) for (int i = 0; i < tokens; ++i) {
                float maximum = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < tokens; ++j) {
                    float score = 0;
                    for (int z = 0; z < d; ++z) score += q[start + i][h * d + z] * k[start + j][h * d + z];
                    score *= scale;
                    if (mask && (*mask)[j] == 0) score = -1e30f;
                    if (window) {
                        const int pos = (i / window - j / window + window - 1) * (2 * window - 1) + i % window - j % window + window - 1;
                        score += scalar(bias, static_cast<size_t>(pos) * heads + h);
                        if (shift) {
                            const int r = win / windows_per_row * window, c = win % windows_per_row * window;
                            if (region(r + i / window, c + i % window) != region(r + j / window, c + j % window)) score -= 100;
                        }
                    }
                    scores[j] = score; maximum = std::max(maximum, score);
                }
                float sum = 0;
                for (float &s : scores) { s = std::exp(s - maximum); sum += s; }
                for (int j = 0; j < tokens; ++j) {
                    const float p = scores[j] / sum;
                    for (int z = 0; z < d; ++z) result[start + i][h * d + z] += p * v[start + j][h * d + z];
                }
            }
        }
        return result;
    }
    std::vector<float> project(const Matrix &pooled, const std::string &prefix) const {
        auto h = linear(pooled, prefix + ".linear1");
        for (float &v : h.data) v = std::max(v, 0.0f);
        auto y = linear(h, prefix + ".linear2");
        double square = 0;
        for (float v : y.data) square += v * v;
        check(std::isfinite(square) && square > 0, "invalid projection norm");
        const float length = static_cast<float>(std::sqrt(square));
        for (float &v : y.data) v /= length;
        return y.data;
    }
    void encode_piece(const std::string &text, std::vector<int> &ids) const {
        UErrorCode status = U_ZERO_ERROR;
        const auto unicode = icu::UnicodeString::fromUTF8(text);
        std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(unicode, status));
        check(U_SUCCESS(status), "tokenizer regex failed");
        while (matcher->find(status)) {
            std::string piece; matcher->group(status).toUTF8String(piece);
            std::vector<std::string> symbols;
            for (unsigned char b : piece) symbols.push_back(byte_chars[b]);
            while (symbols.size() > 1) {
                int best = std::numeric_limits<int>::max();
                std::pair<std::string, std::string> pair;
                for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                    auto it = ranks.find({symbols[i], symbols[i + 1]});
                    if (it != ranks.end() && it->second < best) { best = it->second; pair = it->first; }
                }
                if (best == std::numeric_limits<int>::max()) break;
                std::vector<std::string> merged;
                for (size_t i = 0; i < symbols.size(); ++i) {
                    if (i + 1 < symbols.size() && symbols[i] == pair.first && symbols[i + 1] == pair.second) {
                        merged.push_back(symbols[i] + symbols[i + 1]);
                        ++i;
                    } else merged.push_back(symbols[i]);
                }
                symbols = std::move(merged);
            }
            for (const auto &symbol : symbols) {
                const auto it = vocab.find(symbol);
                check(it != vocab.end(), "tokenizer has no token for byte sequence"); ids.push_back(it->second);
            }
        }
        check(U_SUCCESS(status), "tokenizer regex failed");
    }
};

Model::Model(const std::string &path, int threads) : impl_(std::make_unique<Impl>()) { impl_->load(path, threads); }
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model &Model::operator=(Model &&) noexcept = default;
int Model::projection_dimension() const { return impl_->config.at("projection_dim"); }

std::vector<int> Model::tokenize(const std::string &text, size_t max_length) const {
    const auto &m = *impl_;
    check(max_length >= 2 && max_length <= static_cast<size_t>(m.t.at("max_position_embeddings").get<int>() - 2), "invalid maximum text length");
    std::vector<int> ids{m.t.at("bos_token_id").get<int>()};
    size_t pos = 0;
    while (pos < text.size()) {
        size_t next = std::string::npos;
        const json *match = nullptr;
        for (const auto &s : m.special) {
            const auto p = text.find(s.at("content").get<std::string>(), pos);
            if (p < next) { next = p; match = &s; }
        }
        if (!match) { m.encode_piece(text.substr(pos), ids); break; }
        size_t end = next;
        // RoBERTa <mask> consumes left whitespace; Unicode handling is required.
        if (match->value("lstrip", false)) {
            auto u = icu::UnicodeString::fromUTF8(text.substr(pos, next - pos));
            while (!u.isEmpty()) {
                const UChar32 cp = u.char32At(u.moveIndex32(u.length(), -1));
                if (!u_isUWhiteSpace(cp)) break;
                u.truncate(u.moveIndex32(u.length(), -1));
            }
            std::string prefix; u.toUTF8String(prefix); end = pos + prefix.size();
        }
        m.encode_piece(text.substr(pos, end - pos), ids);
        ids.push_back(match->at("id").get<int>());
        pos = next + match->at("content").get<std::string>().size();
        if (match->value("rstrip", false)) {
            while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
        }
    }
    if (ids.size() > max_length - 1) ids.resize(max_length - 1);
    ids.push_back(m.t.at("eos_token_id").get<int>());
    return ids;
}

std::vector<float> Model::text(const std::string &utf8_text) { return text(tokenize(utf8_text)); }
std::vector<float> Model::text(const std::vector<int> &ids) {
    auto &m = *impl_;
    const int dim = m.t.at("hidden_size"), pad = m.t.at("pad_token_id"), vocab = m.t.at("vocab_size");
    check(!ids.empty() && ids.size() <= static_cast<size_t>(m.t.at("max_position_embeddings").get<int>() - 2), "text length outside model context");
    const auto *word = m.weight("text.embeddings.word_embeddings.weight"), *position = m.weight("text.embeddings.position_embeddings.weight"),
               *type = m.weight("text.embeddings.token_type_embeddings.weight");
    Matrix x(static_cast<int>(ids.size()), dim);
    std::vector<int> mask(ids.size());
    int counter = pad;
    for (int i = 0; i < x.rows; ++i) {
        check(ids[i] >= 0 && ids[i] < vocab, "token ID out of vocabulary");
        mask[i] = ids[i] != pad;
        const int pos = mask[i] ? ++counter : pad;
        check(pos < m.t.at("max_position_embeddings").get<int>(), "text positions overflow");
        for (int c = 0; c < dim; ++c) x[i][c] = scalar(word, static_cast<size_t>(ids[i]) * dim + c) + scalar(position, static_cast<size_t>(pos) * dim + c) + scalar(type, c);
    }
    check(std::any_of(mask.begin(), mask.end(), [](int x) { return x; }), "text contains only padding");
    const float eps = m.t.at("layer_norm_eps");
    x = m.norm(x, "text.embeddings.LayerNorm", eps);
    for (int l = 0; l < m.t.at("num_hidden_layers").get<int>(); ++l) {
        const auto p = "text.encoder.layer." + std::to_string(l);
        auto y = m.attention(x, p + ".attention.self", m.t.at("num_attention_heads"), x.rows, &mask);
        y = m.linear(y, p + ".attention.output.dense"); add(y, x);
        x = m.norm(y, p + ".attention.output.LayerNorm", eps);
        y = m.linear(x, p + ".intermediate.dense"); gelu(y);
        y = m.linear(y, p + ".output.dense"); add(y, x);
        x = m.norm(y, p + ".output.LayerNorm", eps);
    }
    Matrix first(1, dim); std::copy_n(x[0], dim, first[0]);
    auto pooled = m.linear(first, "text.pooler.dense");
    for (float &v : pooled.data) v = std::tanh(v);
    return m.project(pooled, "text_proj");
}

std::vector<float> Model::audio_features(const std::vector<float> &features, int frames) {
    auto &m = *impl_;
    const int bins = m.a.at("num_mel_bins"), spec = m.a.at("spec_size"), ratio = spec / bins;
    const int target_time = spec * ratio;
    check(frames > 0 && frames <= target_time && features.size() == static_cast<size_t>(frames) * bins, "invalid mel feature shape");
    check(std::all_of(features.begin(), features.end(), [](float x) { return std::isfinite(x); }), "non-finite mel features");
    Matrix mel(frames, bins);
    const auto *bnw = m.weight("audio.batch_norm.weight"), *bnb = m.weight("audio.batch_norm.bias"),
               *mean = m.weight("audio.batch_norm.running_mean"), *variance = m.weight("audio.batch_norm.running_var");
    for (int t = 0; t < frames; ++t) for (int f = 0; f < bins; ++f)
        mel[t][f] = (features[static_cast<size_t>(t) * bins + f] - scalar(mean, f)) / std::sqrt(scalar(variance, f) + 1e-5f) * scalar(bnw, f) + scalar(bnb, f);
    Matrix resized(target_time, bins);
    if (frames == target_time) resized = std::move(mel);
    else for (int t = 0; t < target_time; ++t) {
        const float source = static_cast<float>(t) * (frames - 1) / (target_time - 1);
        const int base = static_cast<int>(std::floor(source));
        const float frac = source - base;
        float coeff[4]; for (int k = 0; k < 4; ++k) coeff[k] = cubic(frac - (k - 1));
        for (int f = 0; f < bins; ++f) for (int k = 0; k < 4; ++k)
            resized[t][f] += mel[std::max(0, std::min(frames - 1, base + k - 1))][f] * coeff[k];
    }
    // mel2img: [ratio, time/ratio, bins] -> [ratio, bins, time/ratio]
    // -> [spec,spec]. Patch convolution lowered to one ggml matrix multiply.
    const int patchdim = m.a.at("patch_embeds_hidden_size");
    int grid = spec / 4;
    Matrix patches(grid * grid, 16);
    for (int r = 0; r < grid; ++r) for (int c = 0; c < grid; ++c)
        for (int y = 0; y < 4; ++y) for (int z = 0; z < 4; ++z) {
            const int image_row = r * 4 + y, image_col = c * 4 + z;
            const int time = (image_row / bins) * spec + image_col, freq = image_row % bins;
            patches[r * grid + c][y * 4 + z] = resized[time][freq];
        }
    // The stored Conv2d [out,1,4,4] has the same contiguous bytes as [out,16].
    const auto *conv = m.weight("audio.patch_embed.proj.weight");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> convctx(ggml_init({4096, nullptr, true}), ggml_free);
    check(convctx != nullptr, "conv metadata allocation failed");
    auto *flat = ggml_new_tensor_2d(convctx.get(), conv->type, 16, patchdim);
    flat->data = conv->data;
    // Use a temporary alias name only in this instance's metadata context.
    // Avoid changing the immutable stored tensor: linear accepts a matrix view.
    auto conv_linear = [&](const Matrix &input) {
        const size_t bytes = (input.data.size() + static_cast<size_t>(input.rows) * patchdim) * sizeof(float) + 2 * 1024 * 1024;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
        check(ctx != nullptr, "patch allocation failed");
        auto *in = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, input.rows);
        std::memcpy(in->data, input.data.data(), input.data.size() * sizeof(float));
        auto *out = ggml_mul_mat(ctx.get(), flat, in);
        auto *graph = ggml_new_graph_custom(ctx.get(), 32, false); ggml_build_forward_expand(graph, out);
        check(ggml_graph_compute_with_ctx(ctx.get(), graph, m.threads) == GGML_STATUS_SUCCESS, "patch convolution failed");
        Matrix result(input.rows, patchdim);
        std::memcpy(result.data.data(), out->data, result.data.size() * sizeof(float));
        const auto *bias = m.weight("audio.patch_embed.proj.bias");
        for (int r = 0; r < result.rows; ++r) for (int c = 0; c < patchdim; ++c) result[r][c] += scalar(bias, c);
        return result;
    };
    Matrix x = m.norm(conv_linear(patches), "audio.patch_embed.norm", 1e-5f);
    const float eps = m.a.at("layer_norm_eps");
    for (int stage = 0; stage < 4; ++stage) {
        const int depth = m.a.at("depths")[stage], heads = m.a.at("num_attention_heads")[stage], window = m.a.at("window_size");
        const int padded = (grid + window - 1) / window * window;
        for (int block = 0; block < depth; ++block) {
            const auto p = "audio.layers." + std::to_string(stage) + ".blocks." + std::to_string(block);
            const int shift = grid <= window ? 0 : (block % 2) * (window / 2);
            auto normalized = m.norm(x, p + ".layernorm_before", eps);
            Matrix windows(padded * padded, x.cols);
            // Partition a cyclically shifted, zero-padded image into windows.
            for (int r = 0; r < padded; ++r) for (int c = 0; c < padded; ++c) {
                const int sr = (r + shift) % padded, sc = (c + shift) % padded;
                const int index = ((r / window) * (padded / window) + c / window) * window * window + (r % window) * window + c % window;
                if (sr < grid && sc < grid) std::copy_n(normalized[sr * grid + sc], x.cols, windows[index]);
            }
            auto attended = m.attention(windows, p + ".attention.self", heads, window * window, nullptr, window, shift, padded);
            attended = m.linear(attended, p + ".attention.output.dense");
            Matrix y(x.rows, x.cols);
            for (int r = 0; r < grid; ++r) for (int c = 0; c < grid; ++c) {
                const int sr = (r - shift + padded) % padded, sc = (c - shift + padded) % padded;
                const int index = ((sr / window) * (padded / window) + sc / window) * window * window + (sr % window) * window + sc % window;
                std::copy_n(attended[index], x.cols, y[r * grid + c]);
            }
            add(x, y);
            y = m.linear(m.norm(x, p + ".layernorm_after", eps), p + ".intermediate.dense"); gelu(y);
            y = m.linear(y, p + ".output.dense"); add(x, y);
        }
        if (stage < 3) {
            check(grid % 2 == 0, "odd stage grid not supported");
            Matrix merged((grid / 2) * (grid / 2), x.cols * 4);
            for (int r = 0; r < grid / 2; ++r) for (int c = 0; c < grid / 2; ++c)
                for (int part = 0; part < 4; ++part) {
                    const int sr = 2 * r + part % 2, sc = 2 * c + part / 2;
                    std::copy_n(x[sr * grid + sc], x.cols, merged[r * (grid / 2) + c] + part * x.cols);
                }
            const auto p = "audio.layers." + std::to_string(stage) + ".downsample";
            x = m.linear(m.norm(merged, p + ".norm", 1e-5f), p + ".reduction", false);
            grid /= 2;
        }
    }
    x = m.norm(x, "audio.norm", 1e-5f);
    Matrix pooled(1, x.cols);
    for (int r = 0; r < x.rows; ++r) for (int c = 0; c < x.cols; ++c) pooled[0][c] += x[r][c] / x.rows;
    return m.project(pooled, "audio_proj");
}
std::vector<float> Model::audio(const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
    check(impl_->a.at("num_mel_bins") == 64, "waveform frontend requires 64 mel bins");
    return audio_features(log_mel(pcm, sample_rate, crop_offset), 1001);
}
} // namespace clap
