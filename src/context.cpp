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

std::vector<int> tokenize(const Model &model, const std::string &utf8, size_t max_length) {
    if (model.vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text tokenizer");
    return model.vocabulary->tokenize(utf8, max_length);
}
std::vector<float> encode_text(Context &context, const std::string &utf8) {
    if (context.model().vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text encoder");
    return context.model().architecture->text(
        context, tokenize(context.model(), utf8, context.model().architecture->text_parameters().context_length));
}
std::vector<float> encode_audio(Context &context, const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
    if (context.model().architecture->has_waveform_embeddings()) {
        if (sample_rate <= 0)
            throw std::runtime_error("CLAP: audio sample rate must be positive");
        const int target = context.model().architecture->sample_rate();
        if (sample_rate == target)
            return context.model().architecture->waveform_embedding(context, pcm, crop_offset);
        return context.model().architecture->waveform_embedding(
            context, resample_audio({pcm, sample_rate}, target).samples, crop_offset);
    }
    if (context.model().architecture->is_waveform_encoder()) {
        if (crop_offset != 0)
            throw std::runtime_error(
                "CLAP: waveform encoders process the complete waveform; crop offsets are unsupported");
        const auto sequence = audio_frames(context, pcm, sample_rate);
        std::vector<float> pooled(sequence.dimension);
        for (int frame = 0; frame < sequence.frames; ++frame)
            for (int column = 0; column < sequence.dimension; ++column)
                pooled[column] +=
                    sequence.values[static_cast<size_t>(frame) * sequence.dimension + column] / sequence.frames;
        return pooled;
    }
    throw std::runtime_error("CLAP: this model has no waveform audio encoder");
}

std::vector<float> encode_audio_file(Context &context, const std::string &path, size_t crop_offset) {
    const auto audio = decode_audio(path);
    return encode_audio(context, audio.samples, audio.sample_rate, crop_offset);
}

AudioFrames audio_frames(Context &context, const std::vector<float> &pcm, int sample_rate) {
    if (!(context.model().architecture->is_waveform_encoder()))
        throw std::runtime_error("CLAP: this model does not expose frame representations");
    if (!(sample_rate > 0))
        throw std::runtime_error("CLAP: audio sample rate must be positive");
    const int target = context.model().architecture->sample_rate();
    if (sample_rate == target)
        return context.model().architecture->audio_frames(context, pcm);
    return context.model().architecture->audio_frames(context, resample_audio({pcm, sample_rate}, target).samples);
}
} // namespace clap::detail
