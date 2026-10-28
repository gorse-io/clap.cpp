#include "clap.h"
#include "model.h"
#include "model-loader.h"
#include "models/models.h"
#include "vocab.h"
#include <stdexcept>

namespace clap {

struct CLAP::Impl {
    // Declaration order ensures the architecture is destroyed before its mapping.
    detail::ModelLoader loader;
    std::unique_ptr<detail::ModelArchitecture> architecture;
    std::unique_ptr<detail::Vocabulary> vocabulary;

    Impl(const std::string &path, int threads) {
        loader.load(path);
        const auto name = loader.metadata("general.architecture");
        if (name == "clap") architecture = std::make_unique<detail::ClapModel>(loader, threads);
        else if (name == "clamp3") architecture = std::make_unique<detail::Clamp3Model>(loader, threads);
        else if (name == "mert2") architecture = std::make_unique<detail::Mert2Model>(loader, threads);
        else if (name == "mert") architecture = std::make_unique<detail::MertModel>(loader, threads);
        else if (name == "muq_mulan") architecture = std::make_unique<detail::MuqMulanModel>(loader, threads);
        else if (name == "muq") architecture = std::make_unique<detail::MuqModel>(loader, threads);
        else if (name == "gemma-embedding2") architecture = std::make_unique<detail::EmbeddingGemma2Model>(loader, threads);
        else throw std::runtime_error("CLAP: unsupported GGUF architecture: " + name);
        loader.validate_tensor_coverage();
        if (architecture->has_text_encoder()) {
            const auto text = architecture->text_parameters();
            if (name == "gemma-embedding2") {
                nlohmann::json tokenizer = nlohmann::json::object();
                for (const auto *key : {"tokenizer.ggml.model", "tokenizer.ggml.tokens",
                        "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
                        "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                        "tokenizer.ggml.unknown_token_id", "tokenizer.ggml.add_bos_token",
                        "tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_space_prefix"})
                    if (loader.has_metadata(key)) tokenizer[key] = loader.metadata_value(key);
                vocabulary = std::make_unique<detail::GemmaVocabulary>(tokenizer, text.context_length);
            } else {
                const auto tokenizer = nlohmann::json::parse(loader.metadata((name + ".tokenizer").c_str()));
                if (name == "clamp3" || name == "muq_mulan")
                    vocabulary = std::make_unique<detail::UnigramVocabulary>(tokenizer, text.bos, text.eos, 3, text.context_length);
                else
                    vocabulary = std::make_unique<detail::ByteBpeVocabulary>(tokenizer, text.bos, text.eos, text.context_length);
            }
        }
    }
};

CLAP::CLAP(const std::string &path, int threads) : impl_(std::make_unique<Impl>(path, threads)) {}
CLAP::~CLAP() = default;
CLAP::CLAP(CLAP &&) noexcept = default;
CLAP &CLAP::operator=(CLAP &&) noexcept = default;

int CLAP::projection_dimension() const { return impl_->architecture->projection_dimension(); }
std::vector<int> CLAP::tokenize(const std::string &utf8, size_t max_length) const {
    if (impl_->vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text tokenizer");
    return impl_->vocabulary->tokenize(utf8, max_length);
}
std::vector<float> CLAP::encode_text(const std::string &utf8) {
    if (impl_->vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text encoder");
    return encode_text(tokenize(utf8, impl_->architecture->text_parameters().context_length));
}
std::vector<float> CLAP::encode_text(const std::vector<int> &ids) {
    return impl_->architecture->text(ids);
}
std::vector<float> CLAP::encode_audio(const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
    if (impl_->architecture->has_waveform_embeddings()) {
        if (sample_rate <= 0) throw std::runtime_error("CLAP: audio sample rate must be positive");
        const int target = impl_->architecture->sample_rate();
        if (sample_rate == target) return impl_->architecture->waveform_embedding(pcm, crop_offset);
        return impl_->architecture->waveform_embedding(resample_audio({pcm, sample_rate}, target).samples, crop_offset);
    }
    if (impl_->architecture->is_waveform_encoder()) {
        if (crop_offset != 0)
            throw std::runtime_error("CLAP: waveform encoders process the complete waveform; crop offsets are unsupported");
        const auto sequence = audio_frames(pcm, sample_rate);
        std::vector<float> pooled(sequence.dimension);
        for (int frame = 0; frame < sequence.frames; ++frame)
            for (int column = 0; column < sequence.dimension; ++column)
                pooled[column] += sequence.values[static_cast<size_t>(frame) * sequence.dimension + column] / sequence.frames;
        return pooled;
    }
    throw std::runtime_error("CLAP: this model has no waveform audio encoder");
}

AudioFrames CLAP::audio_frames(const std::vector<float> &pcm, int sample_rate) {
    if (!(impl_->architecture->is_waveform_encoder()))
        throw std::runtime_error("CLAP: this model does not expose frame representations");
    if (!(sample_rate > 0))
        throw std::runtime_error("CLAP: audio sample rate must be positive");
    const int target = impl_->architecture->sample_rate();
    if (sample_rate == target) return impl_->architecture->audio_frames(pcm);
    return impl_->architecture->audio_frames(resample_audio({pcm, sample_rate}, target).samples);
}

int CLAP::audio_sample_rate() const { return impl_->architecture->sample_rate(); }
AudioFrames CLAP::clamp3_features(const std::vector<float> &pcm, int sample_rate) {
    if (sample_rate <= 0) throw std::runtime_error("CLAP: audio sample rate must be positive");
    if (sample_rate == 24000) return impl_->architecture->clamp3_features(pcm);
    return impl_->architecture->clamp3_features(resample_audio({pcm, sample_rate}, 24000).samples);
}
} // namespace clap
