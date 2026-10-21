#include "clap.h"
#include "model.h"
#include "model-loader.h"
#include "models/models.h"
#include "vocab.h"
#include <stdexcept>

namespace clap {

struct Model::Impl {
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
        else throw std::runtime_error("CLAP: unsupported GGUF architecture: " + name);
        if (name != "mert2" && name != "mert") {
            const auto text = architecture->text_parameters();
            const auto tokenizer = nlohmann::json::parse(loader.metadata((name + ".tokenizer").c_str()));
            if (name == "clamp3")
                vocabulary = std::make_unique<detail::UnigramVocabulary>(tokenizer, text.bos, text.eos, 3, text.context_length);
            else
                vocabulary = std::make_unique<detail::ByteBpeVocabulary>(tokenizer, text.bos, text.eos, text.context_length);
        }
    }
};

Model::Model(const std::string &path, int threads) : impl_(std::make_unique<Impl>(path, threads)) {}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model &Model::operator=(Model &&) noexcept = default;

int Model::projection_dimension() const { return impl_->architecture->projection_dimension(); }
std::vector<int> Model::tokenize(const std::string &utf8, size_t max_length) const {
    if (impl_->vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text tokenizer");
    return impl_->vocabulary->tokenize(utf8, max_length);
}
std::vector<float> Model::text(const std::string &utf8) {
    if (impl_->vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text encoder");
    return text(tokenize(utf8, impl_->architecture->text_parameters().context_length));
}
std::vector<float> Model::text(const std::vector<int> &ids) {
    return impl_->architecture->text(ids);
}
std::vector<float> Model::audio_features(const std::vector<float> &features, int frames) {
    return impl_->architecture->audio_features(features, frames);
}
std::vector<float> Model::audio(const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
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
    if (impl_->architecture->mel_bins() == 0)
        throw std::runtime_error("CLAP: CLaMP3 waveform input is unavailable; extract MERT-v1-95M features and use audio_features/--audio-features");
    if (impl_->architecture->mel_bins() != 64)
        throw std::runtime_error("CLAP: waveform frontend requires 64 mel bins");
    return audio_features(log_mel(pcm, sample_rate, crop_offset), 1001);
}

AudioFrames Model::audio_frames(const std::vector<float> &pcm, int sample_rate) {
    if (!(impl_->architecture->is_waveform_encoder()))
        throw std::runtime_error("CLAP: this model does not expose frame representations");
    if (!(sample_rate > 0))
        throw std::runtime_error("CLAP: audio sample rate must be positive");
    const int target = impl_->architecture->sample_rate();
    if (sample_rate == target) return impl_->architecture->audio_frames(pcm);
    return impl_->architecture->audio_frames(resample_audio({pcm, sample_rate}, target).samples);
}

int Model::audio_sample_rate() const { return impl_->architecture->sample_rate(); }
AudioFrames Model::clamp3_features(const std::vector<float> &pcm, int sample_rate) {
    if (!(sample_rate > 0)) throw std::runtime_error("audio sample rate must be positive");
    if (sample_rate == 24000) return impl_->architecture->clamp3_features(pcm);
    return impl_->architecture->clamp3_features(resample_audio({pcm, sample_rate}, 24000).samples);
}
} // namespace clap
