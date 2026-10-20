#include "models.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace clap::detail {
using json = nlohmann::json;

Clamp3Model::Clamp3Model(const ModelLoader &source, int threads)
    : loader(source), ops(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
}

void Clamp3Model::load_arch_hparams() {
    config = json::parse(loader.metadata("clamp3.config"));
    check(config.at("model_type") == "clamp3", "not a CLaMP3 model");
    check(config.at("chunk_length").get<int>() == 128, "CLaMP3 chunk length must be 128");
    audio_config = config.at("audio_config");
    text_config = config.at("text_config");
    for (const auto *key : {"hidden_size", "num_hidden_layers", "num_attention_heads",
                            "intermediate_size", "max_position_embeddings", "type_vocab_size"}) {
        check(audio_config.at(key).get<int>() > 0, std::string("invalid audio ") + key);
        check(text_config.at(key).get<int>() > 0, std::string("invalid text ") + key);
    }
    check(audio_config.at("hidden_size") == config.at("audio_feature_dim"), "audio feature dimension mismatch");
    check(audio_config.at("max_position_embeddings").get<int>() >= 128, "audio position table is too short");
    check(text_config.at("max_position_embeddings").get<int>() >= 130, "text position table is too short");
    check(audio_config.at("hidden_size").get<int>() % audio_config.at("num_attention_heads").get<int>() == 0,
          "invalid audio attention dimensions");
    check(text_config.at("hidden_size").get<int>() % text_config.at("num_attention_heads").get<int>() == 0,
          "invalid text attention dimensions");
    check(audio_config.value("hidden_act", std::string("gelu")) == "gelu" &&
          text_config.value("hidden_act", std::string("gelu")) == "gelu", "only exact GELU is supported");
    check(text_config.value("position_embedding_type", std::string("absolute")) == "absolute" &&
          audio_config.value("position_embedding_type", std::string("absolute")) == "absolute",
          "only absolute position embeddings are supported");
    const int pad = text_config.at("pad_token_id"), vocab = text_config.at("vocab_size");
    check(pad >= 0 && pad < vocab && pad <= text_config.at("max_position_embeddings").get<int>() - 129,
          "invalid text padding token");
    for (const auto *key : {"bos_token_id", "eos_token_id"})
        check(text_config.at(key).get<int>() >= 0 && text_config.at(key).get<int>() < vocab,
              "invalid text special token");
    for (const auto *branch : {&audio_config, &text_config}) {
        const float epsilon = branch->at("layer_norm_eps").get<float>();
        check(std::isfinite(epsilon) && epsilon > 0, "invalid layer norm epsilon");
        check(!branch->value("is_decoder", false) && !branch->value("add_cross_attention", false),
              "CLaMP3 decoder and cross-attention configurations are unsupported");
    }
    check(config.at("projection_dim").get<int>() > 0, "invalid projection dimension");
}

void Clamp3Model::load_arch_tensors() const {
    auto validate = [&](const json &c, const std::string &prefix, bool word_embeddings) {
        const int dim = c.at("hidden_size"), layers = c.at("num_hidden_layers");
        if (word_embeddings)
            loader.shape(prefix + ".embeddings.word_embeddings.weight", {dim, c.at("vocab_size").get<int>()});
        loader.shape(prefix + ".embeddings.position_embeddings.weight",
                     {dim, c.at("max_position_embeddings").get<int>()});
        loader.shape(prefix + ".embeddings.token_type_embeddings.weight", {dim, c.at("type_vocab_size").get<int>()});
        loader.norm_shape(prefix + ".embeddings.LayerNorm", dim);
        for (int layer = 0; layer < layers; ++layer) {
            const auto p = prefix + ".encoder.layer." + std::to_string(layer);
            for (const auto *name : {"query", "key", "value"})
                loader.linear_shape(p + ".attention.self." + name, dim, dim);
            loader.linear_shape(p + ".attention.output.dense", dim, dim);
            loader.norm_shape(p + ".attention.output.LayerNorm", dim);
            loader.linear_shape(p + ".intermediate.dense", dim, c.at("intermediate_size"));
            loader.linear_shape(p + ".output.dense", c.at("intermediate_size"), dim);
            loader.norm_shape(p + ".output.LayerNorm", dim);
        }
    };
    validate(text_config, "text", true);
    validate(audio_config, "audio", false);
    loader.linear_shape("text_proj", text_config.at("hidden_size"), config.at("projection_dim"));
    loader.linear_shape("audio_proj", audio_config.at("hidden_size"), config.at("projection_dim"));
}

TextParameters Clamp3Model::text_parameters() const {
    return {text_config.at("bos_token_id"), text_config.at("eos_token_id"),
            std::numeric_limits<size_t>::max()};
}
int Clamp3Model::projection_dimension() const { return config.at("projection_dim"); }
int Clamp3Model::mel_bins() const { return 0; }

Matrix Clamp3Model::attention(const Matrix &x, const std::vector<int> &mask,
                              const std::string &prefix, int heads) const {
    const auto q = ops.linear(x, prefix + ".query");
    const auto k = ops.linear(x, prefix + ".key");
    const auto v = ops.linear(x, prefix + ".value");
    Matrix result(x.rows, x.cols);
    const int head_dim = x.cols / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(x.rows);
    for (int head = 0; head < heads; ++head) for (int i = 0; i < x.rows; ++i) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < x.rows; ++j) {
            float score = 0;
            for (int d = 0; d < head_dim; ++d)
                score += q[i][head * head_dim + d] * k[j][head * head_dim + d];
            score = mask[j] ? score * scale : -1e30f;
            scores[j] = score;
            maximum = std::max(maximum, score);
        }
        float sum = 0;
        for (float &score : scores) { score = std::exp(score - maximum); sum += score; }
        for (int j = 0; j < x.rows; ++j) {
            const float probability = scores[j] / sum;
            for (int d = 0; d < head_dim; ++d)
                result[i][head * head_dim + d] += probability * v[j][head * head_dim + d];
        }
    }
    return result;
}

Matrix Clamp3Model::encode(const Matrix &embeddings, const std::vector<int> &mask,
                           const json &branch, const std::string &prefix) const {
    Matrix x = ops.norm(embeddings, prefix + ".embeddings.LayerNorm", branch.at("layer_norm_eps"));
    for (int layer = 0; layer < branch.at("num_hidden_layers").get<int>(); ++layer) {
        const auto p = prefix + ".encoder.layer." + std::to_string(layer);
        auto y = attention(x, mask, p + ".attention.self", branch.at("num_attention_heads"));
        y = ops.linear(y, p + ".attention.output.dense");
        add(y, x);
        x = ops.norm(y, p + ".attention.output.LayerNorm", branch.at("layer_norm_eps"));
        y = ops.linear(x, p + ".intermediate.dense");
        gelu(y);
        y = ops.linear(y, p + ".output.dense");
        add(y, x);
        x = ops.norm(y, p + ".output.LayerNorm", branch.at("layer_norm_eps"));
    }
    return x;
}

std::vector<float> Clamp3Model::pool_project(const Matrix &hidden, const std::vector<int> &mask,
                                              const std::string &projection) const {
    int count = 0;
    Matrix pooled(1, hidden.cols);
    for (int r = 0; r < hidden.rows; ++r) if (mask[r]) {
        ++count;
        for (int c = 0; c < hidden.cols; ++c) pooled[0][c] += hidden[r][c];
    }
    check(count > 0, "CLaMP3 chunk contains no input");
    for (float &value : pooled.data) value /= count;
    return ops.linear(pooled, projection).data;
}

std::vector<float> Clamp3Model::chunked_text(const std::vector<int> &ids) const {
    const int chunk = config.at("chunk_length"), dim = text_config.at("hidden_size");
    const int pad = text_config.at("pad_token_id"), vocab = text_config.at("vocab_size");
    const auto *words = loader.weight("text.embeddings.word_embeddings.weight");
    const auto *positions = loader.weight("text.embeddings.position_embeddings.weight");
    const auto *types = loader.weight("text.embeddings.token_type_embeddings.weight");
    std::vector<float> result(projection_dimension());
    std::vector<float> position_row(dim), type_row(dim);
    tensor_row(types, 0, type_row.data());
    const size_t segments = (ids.size() + chunk - 1) / chunk;
    const size_t remainder = ids.size() % chunk;
    double total_weight = 0;
    for (size_t part = 0; part < segments; ++part) {
        const size_t start = part + 1 == segments && ids.size() > static_cast<size_t>(chunk)
                           ? ids.size() - chunk : part * chunk;
        const size_t length = std::min(static_cast<size_t>(chunk), ids.size() - start);
        Matrix embeddings(chunk, dim);
        std::vector<int> mask(chunk);
        int position = pad;
        for (int row = 0; row < chunk; ++row) {
            const int id = static_cast<size_t>(row) < length ? ids[start + row] : pad;
            check(id >= 0 && id < vocab, "token ID out of vocabulary");
            mask[row] = static_cast<size_t>(row) < length;
            const int pos = id == pad ? pad : ++position;
            check(pos < text_config.at("max_position_embeddings").get<int>(), "text positions overflow");
            tensor_row(words, id, embeddings[row]);
            tensor_row(positions, pos, position_row.data());
            for (int c = 0; c < dim; ++c) embeddings[row][c] = embeddings[row][c] + position_row[c] + type_row[c];
        }
        const auto projected = pool_project(encode(embeddings, mask, text_config, "text"), mask, "text_proj");
        const size_t weight = part + 1 == segments && remainder ? remainder : chunk;
        total_weight += weight;
        for (size_t i = 0; i < result.size(); ++i) result[i] += projected[i] * weight;
    }
    for (float &value : result) value = static_cast<float>(value / total_weight);
    return result;
}

std::vector<float> Clamp3Model::chunked_audio(const Matrix &features) const {
    const int chunk = config.at("chunk_length"), dim = audio_config.at("hidden_size");
    const auto *positions = loader.weight("audio.embeddings.position_embeddings.weight");
    const auto *types = loader.weight("audio.embeddings.token_type_embeddings.weight");
    std::vector<float> result(projection_dimension());
    std::vector<float> position_row(dim), type_row(dim);
    tensor_row(types, 0, type_row.data());
    const size_t frames = features.rows;
    const size_t segments = (frames + chunk - 1) / chunk;
    const size_t remainder = frames % chunk;
    double total_weight = 0;
    for (size_t part = 0; part < segments; ++part) {
        const size_t start = part + 1 == segments && frames > static_cast<size_t>(chunk)
                           ? frames - chunk : part * chunk;
        const size_t length = std::min(static_cast<size_t>(chunk), frames - start);
        Matrix embeddings(chunk, dim);
        std::vector<int> mask(chunk);
        for (int row = 0; row < chunk; ++row) {
            mask[row] = static_cast<size_t>(row) < length;
            tensor_row(positions, row, position_row.data());
            for (int c = 0; c < dim; ++c) {
                const float input = mask[row] ? features[static_cast<int>(start) + row][c] : 0.0f;
                embeddings[row][c] = input + position_row[c] + type_row[c];
            }
        }
        const auto projected = pool_project(encode(embeddings, mask, audio_config, "audio"), mask, "audio_proj");
        const size_t weight = part + 1 == segments && remainder ? remainder : chunk;
        total_weight += weight;
        for (size_t i = 0; i < result.size(); ++i) result[i] += projected[i] * weight;
    }
    for (float &value : result) value = static_cast<float>(value / total_weight);
    return result;
}

std::vector<float> Clamp3Model::text(const std::vector<int> &ids) const {
    check(!ids.empty(), "text token sequence is empty");
    return chunked_text(ids);
}

std::vector<float> Clamp3Model::audio_features(const std::vector<float> &features, int frames) const {
    const int dim = config.at("audio_feature_dim");
    check(frames > 0 && frames <= std::numeric_limits<int>::max() - 2 &&
          features.size() == static_cast<size_t>(frames) * dim,
          "invalid MERT-v1-95M feature shape");
    check(std::all_of(features.begin(), features.end(), [](float value) { return std::isfinite(value); }),
          "non-finite MERT-v1-95M features");
    Matrix bordered(frames + 2, dim);
    std::copy(features.begin(), features.end(), bordered.data.begin() + dim);
    return chunked_audio(bordered);
}
} // namespace clap::detail
