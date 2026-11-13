/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "context.h"
#include "models.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace clap::detail {
TensorLoadPolicy Clamp3Model::loading_policy(const ModelLoader &loader) {
    TensorLoadPolicy policy;
    if (loader.has_metadata_prefix("clamp3.audio_frontend")) {
        policy.unrecognized_tensor_error = "CLAP: unrecognized model tensor: ";
        policy.validate_tensor_type = [](const ggml_tensor *w) {
            if (std::string(w->name).rfind("mert.", 0) == 0 && w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16 &&
                w->type != GGML_TYPE_BF16)
                throw std::runtime_error("CLAP: MERT supports F32/F16/BF16 weights only");
        };
    }
    return policy;
}

using json = nlohmann::json;

Clamp3Model::Clamp3Model(const ModelLoader &source) : loader(source) {
    load_arch_hparams();
    load_arch_tensors();
    if (loader.has_metadata_prefix("clamp3.audio_frontend")) {
        if (loader.metadata("clamp3.audio_frontend") != "mert-v1")
            throw std::runtime_error("CLAP: unsupported CLaMP3 audio frontend");
        const auto revision = loader.metadata("clamp3.mert.source_revision");
        if (revision.size() != 40 || revision.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::runtime_error("CLAP: invalid bundled MERT source revision");
        const auto mert_config = json::parse(loader.metadata("clamp3.mert.config"));
        if (!mert_config.at("hidden_size").is_number_integer() ||
            mert_config.at("hidden_size") != config.at("audio_feature_dim"))
            throw std::runtime_error("CLAP: CLaMP3/MERT feature dimension mismatch");
        frontend = std::make_unique<MertModel>(loader, "mert.", "clamp3.");
    } else {
        throw std::runtime_error(
            "CLAP: CLaMP3 requires bundled MERT-v1-95M audio frontend metadata; reconvert the checkpoint");
    }
}
Clamp3Model::~Clamp3Model() = default;
void Clamp3Model::initialize_context(Context &context) const {
    context.runtime(this, loader, false);
    if (frontend)
        frontend->initialize_context(context);
}

bool Clamp3Model::has_waveform_embeddings() const { return frontend != nullptr; }
std::vector<float> Clamp3Model::waveform_embedding(Context &context, const std::vector<float> &pcm,
                                                   size_t crop_offset) const {
    if (crop_offset != 0)
        throw std::runtime_error("CLAP: crop offsets are unsupported for this model");
    if (!frontend)
        throw std::runtime_error("CLAP: CLaMP3 requires MERT-v1-95M features");
    const auto features = frontend->clamp3_features(context, pcm);
    return encode_features(context, features.values, features.frames);
}

void Clamp3Model::load_arch_hparams() {
    config = json::parse(loader.metadata("clamp3.config"));
    if (config.at("model_type") != "clamp3")
        throw std::runtime_error("CLAP: not a CLaMP3 model");
    if (config.at("chunk_length").get<int>() != 128)
        throw std::runtime_error("CLAP: CLaMP3 chunk length must be 128");
    audio_config = config.at("audio_config");
    text_config = config.at("text_config");
    for (const auto *key : {"hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                            "max_position_embeddings", "type_vocab_size"}) {
        if (!(audio_config.at(key).get<int>() > 0))
            throw std::runtime_error(std::string("CLAP: invalid audio ") + key);
        if (!(text_config.at(key).get<int>() > 0))
            throw std::runtime_error(std::string("CLAP: invalid text ") + key);
    }
    if (audio_config.at("hidden_size") != config.at("audio_feature_dim"))
        throw std::runtime_error("CLAP: audio feature dimension mismatch");
    if (!(audio_config.at("max_position_embeddings").get<int>() >= 128))
        throw std::runtime_error("CLAP: audio position table is too short");
    if (!(text_config.at("max_position_embeddings").get<int>() >= 130))
        throw std::runtime_error("CLAP: text position table is too short");
    if (audio_config.at("hidden_size").get<int>() % audio_config.at("num_attention_heads").get<int>() != 0)
        throw std::runtime_error("CLAP: invalid audio attention dimensions");
    if (text_config.at("hidden_size").get<int>() % text_config.at("num_attention_heads").get<int>() != 0)
        throw std::runtime_error("CLAP: invalid text attention dimensions");
    if (!(audio_config.value("hidden_act", std::string("gelu")) == "gelu" &&
          text_config.value("hidden_act", std::string("gelu")) == "gelu"))
        throw std::runtime_error("CLAP: only exact GELU is supported");
    if (!(text_config.value("position_embedding_type", std::string("absolute")) == "absolute" &&
          audio_config.value("position_embedding_type", std::string("absolute")) == "absolute"))
        throw std::runtime_error("CLAP: only absolute position embeddings are supported");
    const int pad = text_config.at("pad_token_id"), vocab = text_config.at("vocab_size");
    if (!(pad >= 0 && pad < vocab && pad <= text_config.at("max_position_embeddings").get<int>() - 129))
        throw std::runtime_error("CLAP: invalid text padding token");
    for (const auto *key : {"bos_token_id", "eos_token_id"})
        if (!(text_config.at(key).get<int>() >= 0 && text_config.at(key).get<int>() < vocab))
            throw std::runtime_error("CLAP: invalid text special token");
    for (const auto *branch : {&audio_config, &text_config}) {
        const float epsilon = branch->at("layer_norm_eps").get<float>();
        if (!(std::isfinite(epsilon) && epsilon > 0))
            throw std::runtime_error("CLAP: invalid layer norm epsilon");
        if (!(!branch->value("is_decoder", false) && !branch->value("add_cross_attention", false)))
            throw std::runtime_error("CLAP: CLaMP3 decoder and cross-attention configurations are unsupported");
    }
    if (!(config.at("projection_dim").get<int>() > 0))
        throw std::runtime_error("CLAP: invalid projection dimension");
}

void Clamp3Model::load_arch_tensors() {
    projection_dim = config.at("projection_dim");
    auto create = [&](Branch &branch, const json &c, const std::string &prefix, bool text) {
        branch.hidden = c.at("hidden_size");
        branch.heads = c.at("num_attention_heads");
        branch.epsilon = c.at("layer_norm_eps");
        branch.positions = c.at("max_position_embeddings");
        branch.pad = c.value("pad_token_id", 0);
        branch.vocab = c.value("vocab_size", 0);
        branch.bos = text ? c.at("bos_token_id").get<int>() : 0;
        branch.eos = text ? c.at("eos_token_id").get<int>() : 0;
        // Audio consumes MERT features, not a word table (nor a required audio vocabulary).
        branch.embeddings = create_embeddings(loader, prefix + ".embeddings", branch.hidden, branch.vocab,
                                              branch.positions, c.at("type_vocab_size").get<int>(), text);
        for (int i = 0; i < c.at("num_hidden_layers").get<int>(); ++i)
            branch.layers.push_back(create_bert_layer(loader, prefix + ".encoder.layer." + std::to_string(i),
                                                      branch.hidden, c.at("intermediate_size").get<int>()));
    };
    create(text_branch, text_config, "text", true);
    create(audio_branch, audio_config, "audio", false);
    text_branch.projection = create_linear(loader, "text_proj", text_branch.hidden, projection_dim);
    audio_branch.projection = create_linear(loader, "audio_proj", audio_branch.hidden, projection_dim);
}

TextParameters Clamp3Model::text_parameters() const {
    return {text_branch.bos, text_branch.eos, std::numeric_limits<size_t>::max()};
}
int Clamp3Model::projection_dimension() const { return projection_dim; }
int Clamp3Model::mel_bins() const { return 0; }
int Clamp3Model::sample_rate() const { return frontend ? frontend->sample_rate() : 0; }
bool Clamp3Model::is_waveform_encoder() const { return false; }
AudioFrames Clamp3Model::audio_frames(Context &, const std::vector<float> &) const {
    throw std::runtime_error("CLAP: CLaMP3 requires precomputed MERT-v1-95M features");
}

ggml_tensor *Clamp3Model::attention(Graph &g, ggml_tensor *x, const std::vector<int> &mask,
                                    const AttentionWeights &weights, int heads) const {
    std::vector<float> scores(mask.size());
    for (size_t i = 0; i < mask.size(); ++i)
        scores[i] = mask[i] ? 0.0f : -1e30f;
    return g.attention(g.linear(x, weights.query), g.linear(x, weights.key), g.linear(x, weights.value), heads,
                       g.input(scores, scores.size(), 1));
}

ggml_tensor *Clamp3Model::encode(Graph &g, ggml_tensor *embeddings, const std::vector<int> &mask,
                                 const Branch &branch) const {
    auto *ctx = g.ctx();
    auto *x = g.norm(embeddings, branch.embeddings.norm, branch.epsilon);
    for (const auto &layer : branch.layers) {
        auto *y = attention(g, x, mask, layer.attention, branch.heads);
        y = g.linear(y, layer.attention.output);
        x = g.norm(ggml_add(ctx, y, x), layer.attention_norm, branch.epsilon);
        y = ggml_gelu_erf(ctx, g.linear(x, layer.intermediate));
        y = g.linear(y, layer.output);
        x = g.norm(ggml_add(ctx, y, x), layer.output_norm, branch.epsilon);
    }
    return x;
}

ggml_tensor *Clamp3Model::pool_project(Graph &g, ggml_tensor *hidden, const std::vector<int> &mask,
                                       const LayerWeights &projection) const {
    return g.linear(g.mean_pool(hidden, mask), projection);
}

std::vector<float> Clamp3Model::chunked_text(Context &context, const std::vector<int> &ids) const {
    const auto &runtime = context.runtime(this, loader, false);
    const int chunk = chunk_length;
    const int pad = text_branch.pad, vocab = text_branch.vocab;
    std::vector<float> result(projection_dimension());
    const size_t segments = (ids.size() + chunk - 1) / chunk;
    const size_t remainder = ids.size() % chunk;
    double total_weight = 0;
    for (size_t part = 0; part < segments; ++part) {
        const size_t start =
            part + 1 == segments && ids.size() > static_cast<size_t>(chunk) ? ids.size() - chunk : part * chunk;
        const size_t length = std::min(static_cast<size_t>(chunk), ids.size() - start);
        std::vector<int> mask(chunk), tokens(chunk), positions(chunk);
        int position = pad;
        for (int row = 0; row < chunk; ++row) {
            const int id = static_cast<size_t>(row) < length ? ids[start + row] : pad;
            if (!(id >= 0 && id < vocab))
                throw std::runtime_error("CLAP: token ID out of vocabulary");
            mask[row] = static_cast<size_t>(row) < length;
            const int pos = id == pad ? pad : ++position;
            if (!(pos < text_branch.positions))
                throw std::runtime_error("CLAP: text positions overflow");
            tokens[row] = id;
            positions[row] = pos;
        }
        const auto output = runtime.run_cached(
            "text:" + std::to_string(length), {GraphInput(tokens), GraphInput(positions)}, [&](Graph &g) {
                auto *embeddings = ggml_add(g.ctx(), g.embedding(text_branch.embeddings.word, g.dynamic_input(0)),
                                            g.embedding(text_branch.embeddings.position, g.dynamic_input(1)));
                embeddings =
                    ggml_add(g.ctx(), embeddings, g.embedding(text_branch.embeddings.type, std::vector<int>{0}));
                return std::vector<ggml_tensor *>{
                    pool_project(g, encode(g, embeddings, mask, text_branch), mask, text_branch.projection)};
            });
        const auto &projected = output.front().values;
        const size_t weight = part + 1 == segments && remainder ? remainder : chunk;
        total_weight += weight;
        for (size_t i = 0; i < result.size(); ++i)
            result[i] += projected[i] * weight;
    }
    for (float &value : result)
        value = static_cast<float>(value / total_weight);
    return result;
}

std::vector<float> Clamp3Model::chunked_audio(Context &context, const HostMatrix &features) const {
    const auto &runtime = context.runtime(this, loader, false);
    const int chunk = chunk_length, dim = audio_branch.hidden;
    std::vector<float> result(projection_dimension());
    const size_t frames = features.rows;
    const size_t segments = (frames + chunk - 1) / chunk;
    const size_t remainder = frames % chunk;
    double total_weight = 0;
    for (size_t part = 0; part < segments; ++part) {
        const size_t start =
            part + 1 == segments && frames > static_cast<size_t>(chunk) ? frames - chunk : part * chunk;
        const size_t length = std::min(static_cast<size_t>(chunk), frames - start);
        HostMatrix input_values(chunk, dim);
        std::vector<int> mask(chunk), positions(chunk);
        for (int row = 0; row < chunk; ++row) {
            mask[row] = static_cast<size_t>(row) < length;
            positions[row] = row;
            for (int c = 0; c < dim; ++c) {
                const float input = mask[row] ? features[static_cast<int>(start) + row][c] : 0.0f;
                input_values[row][c] = input;
            }
        }
        const auto output =
            runtime.run_cached("audio:" + std::to_string(length), {GraphInput(input_values)}, [&](Graph &g) {
                auto *embeddings =
                    ggml_add(g.ctx(), g.dynamic_input(0), g.embedding(audio_branch.embeddings.position, positions));
                embeddings =
                    ggml_add(g.ctx(), embeddings, g.embedding(audio_branch.embeddings.type, std::vector<int>{0}));
                return std::vector<ggml_tensor *>{
                    pool_project(g, encode(g, embeddings, mask, audio_branch), mask, audio_branch.projection)};
            });
        const auto &projected = output.front().values;
        const size_t weight = part + 1 == segments && remainder ? remainder : chunk;
        total_weight += weight;
        for (size_t i = 0; i < result.size(); ++i)
            result[i] += projected[i] * weight;
    }
    for (float &value : result)
        value = static_cast<float>(value / total_weight);
    return result;
}

std::vector<float> Clamp3Model::text(Context &context, const std::vector<int> &ids) const {
    if (ids.empty())
        throw std::runtime_error("CLAP: text token sequence is empty");
    return chunked_text(context, ids);
}

std::vector<float> Clamp3Model::encode_features(Context &context, const std::vector<float> &features,
                                                int frames) const {
    const int dim = audio_branch.hidden;
    if (!(frames > 0 && frames <= std::numeric_limits<int>::max() - 2 &&
          features.size() == static_cast<size_t>(frames) * dim))
        throw std::runtime_error("CLAP: invalid MERT-v1-95M feature shape");
    if (!(std::all_of(features.begin(), features.end(), [](float value) { return std::isfinite(value); })))
        throw std::runtime_error("CLAP: non-finite MERT-v1-95M features");
    HostMatrix bordered(frames + 2, dim);
    std::copy(features.begin(), features.end(), bordered.data.begin() + dim);
    return chunked_audio(context, bordered);
}
} // namespace clap::detail
