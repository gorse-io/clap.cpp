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
#include "audio.h"
#include "context.h"
#include "vocab.h"

#include <stdexcept>
#include <utility>

namespace clap::detail {
namespace {
int context_threads(const std::shared_ptr<ModelWeights> &weights, int requested) {
    if (!weights || requested < 0)
        throw std::invalid_argument("CLAP: valid model weights and nonnegative threads are required");
    return requested ? requested : weights->default_threads();
}
} // namespace
Context::Context(std::shared_ptr<const Model> model, int requested)
    : owner(std::move(model)), weights(owner ? owner->weights : nullptr), threads(context_threads(weights, requested)) {
    owner->architecture->initialize_context(*this);
}
Context::Context(std::shared_ptr<ModelWeights> shared_weights, int requested)
    : weights(std::move(shared_weights)), threads(context_threads(weights, requested)) {}
Context::~Context() = default;
const Model &Context::model() const {
    if (!owner)
        throw std::logic_error("CLAP: standalone execution context has no model");
    return *owner;
}
GraphRuntime &Context::runtime(const ModelArchitecture *identity, const ModelLoader &loader, bool preserve) {
    const auto found = runtimes.find(identity);
    if (found != runtimes.end()) {
        if (found->second.loader != &loader || found->second.preserve != preserve)
            throw std::logic_error("CLAP: execution context configuration mismatch");
        return *found->second.execution;
    }
    auto execution = std::make_unique<GraphRuntime>(loader, threads, preserve, weights);
    auto entry = runtimes.emplace(identity, Slot{&loader, preserve, std::move(execution)}).first;
    return *entry->second.execution;
}

std::vector<int> tokenize(const CLAP &model, const std::string &utf8, size_t max_length) {
    if (model.context.model().vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text tokenizer");
    return model.context.model().vocabulary->tokenize(utf8, max_length);
}
std::vector<float> encode_text(CLAP &model, const std::string &utf8) {
    if (model.context.model().vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text encoder");
    return model.context.model().architecture->text(
        model.context, tokenize(model, utf8, model.context.model().architecture->text_parameters().context_length));
}
std::vector<float> encode_audio(CLAP &model, const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
    if (model.context.model().architecture->has_waveform_embeddings()) {
        if (sample_rate <= 0)
            throw std::runtime_error("CLAP: audio sample rate must be positive");
        const int target = model.context.model().architecture->sample_rate();
        if (sample_rate == target)
            return model.context.model().architecture->waveform_embedding(model.context, pcm, crop_offset);
        return model.context.model().architecture->waveform_embedding(
            model.context, resample_audio({pcm, sample_rate}, target).samples, crop_offset);
    }
    if (model.context.model().architecture->is_waveform_encoder()) {
        if (crop_offset != 0)
            throw std::runtime_error(
                "CLAP: waveform encoders process the complete waveform; crop offsets are unsupported");
        const auto sequence = audio_frames(model, pcm, sample_rate);
        std::vector<float> pooled(sequence.dimension);
        for (int frame = 0; frame < sequence.frames; ++frame)
            for (int column = 0; column < sequence.dimension; ++column)
                pooled[column] +=
                    sequence.values[static_cast<size_t>(frame) * sequence.dimension + column] / sequence.frames;
        return pooled;
    }
    throw std::runtime_error("CLAP: this model has no waveform audio encoder");
}

std::vector<float> encode_audio_file(CLAP &model, const std::string &path, size_t crop_offset) {
    const auto audio = decode_audio(path);
    return encode_audio(model, audio.samples, audio.sample_rate, crop_offset);
}

AudioFrames audio_frames(CLAP &model, const std::vector<float> &pcm, int sample_rate) {
    if (!(model.context.model().architecture->is_waveform_encoder()))
        throw std::runtime_error("CLAP: this model does not expose frame representations");
    if (!(sample_rate > 0))
        throw std::runtime_error("CLAP: audio sample rate must be positive");
    const int target = model.context.model().architecture->sample_rate();
    if (sample_rate == target)
        return model.context.model().architecture->audio_frames(model.context, pcm);
    return model.context.model().architecture->audio_frames(model.context,
                                                            resample_audio({pcm, sample_rate}, target).samples);
}
} // namespace clap::detail

CLAP::CLAP(const std::string &path, const clap::RuntimeOptions &options)
    : context(std::make_shared<clap::detail::Model>(path, options), options.threads) {}
CLAP::CLAP(std::shared_ptr<const clap::detail::Model> model, int threads) : context(std::move(model), threads) {}
CLAP::~CLAP() = default;
