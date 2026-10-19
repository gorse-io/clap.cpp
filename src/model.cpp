#include "clap.h"
#include "model.h"
#include "model-loader.h"
#include "impl.h"
#include "models/models.h"
#include "vocab.h"

namespace clap {

struct Model::Impl {
    // Declaration order ensures the architecture is destroyed before its mapping.
    detail::ModelLoader loader;
    std::unique_ptr<detail::ModelArchitecture> architecture;
    std::unique_ptr<detail::ByteBpeVocabulary> vocabulary;

    Impl(const std::string &path, int threads) {
        loader.load(path);
        const auto name = loader.metadata("general.architecture");
        detail::check(name == "clap", "unsupported GGUF architecture");
        architecture = std::make_unique<detail::ClapModel>(loader, threads);
        const auto text = architecture->text_parameters();
        vocabulary = std::make_unique<detail::ByteBpeVocabulary>(
            nlohmann::json::parse(loader.metadata((name + ".tokenizer").c_str())),
            text.bos, text.eos, text.context_length);
    }
};

Model::Model(const std::string &path, int threads) : impl_(std::make_unique<Impl>(path, threads)) {}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model &Model::operator=(Model &&) noexcept = default;

int Model::projection_dimension() const { return impl_->architecture->projection_dimension(); }
std::vector<int> Model::tokenize(const std::string &utf8, size_t max_length) const {
    return impl_->vocabulary->tokenize(utf8, max_length);
}
std::vector<float> Model::text(const std::string &utf8) { return text(tokenize(utf8)); }
std::vector<float> Model::text(const std::vector<int> &ids) {
    return impl_->architecture->text(ids);
}
std::vector<float> Model::audio_features(const std::vector<float> &features, int frames) {
    return impl_->architecture->audio_features(features, frames);
}
std::vector<float> Model::audio(const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
    detail::check(impl_->architecture->mel_bins() == 64, "waveform frontend requires 64 mel bins");
    return audio_features(log_mel(pcm, sample_rate, crop_offset), 1001);
}
} // namespace clap
