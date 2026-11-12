#include "audio.h"
#include "model.h"
#include "models/models.h"
#include "vocab.h"

#include <stdexcept>

CLAP::CLAP(const std::string &path, const clap::RuntimeOptions &options, const std::string &mmproj_path)
    : loader(options) {
    using namespace clap::detail;
    const int threads = options.threads;
    loader.load(path);
    const auto name = loader.metadata("general.architecture");
    if (!mmproj_path.empty()) {
        if (name != "gemma-embedding2")
            throw std::runtime_error("CLAP: mmproj is only supported for EmbeddingGemma2");
        audio_loader = std::make_unique<ModelLoader>(options);
        audio_loader->load(mmproj_path);
    }
    if (name == "qwen3")
        architecture = std::make_unique<Qwen3Model>(loader, threads);
    else if (name == "clap")
        architecture = std::make_unique<ClapModel>(loader, threads);
    else if (name == "clap2")
        architecture = std::make_unique<Clap2Model>(loader, threads);
    else if (name == "clamp3")
        architecture = std::make_unique<Clamp3Model>(loader, threads);
    else if (name == "mert2")
        architecture = std::make_unique<Mert2Model>(loader, threads);
    else if (name == "mert")
        architecture = std::make_unique<MertModel>(loader, threads);
    else if (name == "muq_mulan")
        architecture = std::make_unique<MuqMulanModel>(loader, threads);
    else if (name == "muq")
        architecture = std::make_unique<MuqModel>(loader, threads);
    else if (name == "gemma-embedding2")
        architecture = std::make_unique<EmbeddingGemma2Model>(loader, threads, audio_loader.get());
    else
        throw std::runtime_error("CLAP: unsupported GGUF architecture: " + name);
    loader.validate_tensor_coverage();
    if (architecture->has_text_encoder()) {
        const auto text = architecture->text_parameters();
        if (name == "qwen3") {
            vocabulary = std::make_unique<ByteBpeVocabulary>(loader, text.context_length);
            return;
        }
        if (name == "gemma-embedding2") {
            nlohmann::json tokenizer = nlohmann::json::object();
            for (const auto *key :
                 {"tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
                  "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.unknown_token_id",
                  "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_space_prefix"})
                if (loader.has_metadata(key))
                    tokenizer[key] = loader.metadata_value(key);
            vocabulary = std::make_unique<GemmaVocabulary>(tokenizer, text.context_length);
        } else {
            const auto tokenizer = nlohmann::json::parse(loader.metadata((name + ".tokenizer").c_str()));
            if (name == "clamp3" || name == "muq_mulan")
                vocabulary = std::make_unique<UnigramVocabulary>(tokenizer, text.bos, text.eos, 3, text.context_length);
            else if (name == "clap2")
                vocabulary =
                    std::make_unique<ByteBpeVocabulary>(tokenizer, text.bos, text.eos, text.context_length, true);
            else
                vocabulary = std::make_unique<ByteBpeVocabulary>(tokenizer, text.bos, text.eos, text.context_length);
        }
    }
}
CLAP::~CLAP() = default;

namespace clap::detail {
std::vector<int> tokenize(const CLAP &model, const std::string &utf8, size_t max_length) {
    if (model.vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text tokenizer");
    return model.vocabulary->tokenize(utf8, max_length);
}
std::vector<float> encode_text(CLAP &model, const std::string &utf8) {
    if (model.vocabulary == nullptr)
        throw std::runtime_error("CLAP: this model has no text encoder");
    return model.architecture->text(tokenize(model, utf8, model.architecture->text_parameters().context_length));
}
std::vector<float> encode_audio(CLAP &model, const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
    if (model.architecture->has_waveform_embeddings()) {
        if (sample_rate <= 0)
            throw std::runtime_error("CLAP: audio sample rate must be positive");
        const int target = model.architecture->sample_rate();
        if (sample_rate == target)
            return model.architecture->waveform_embedding(pcm, crop_offset);
        return model.architecture->waveform_embedding(resample_audio({pcm, sample_rate}, target).samples, crop_offset);
    }
    if (model.architecture->is_waveform_encoder()) {
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

std::vector<float> encode_image(CLAP &model, const Image &image) { return model.architecture->image_embedding(image); }

std::vector<float> encode_image_file(CLAP &model, const std::string &path) {
    return encode_image(model, decode_image(path));
}

AudioFrames audio_frames(CLAP &model, const std::vector<float> &pcm, int sample_rate) {
    if (!(model.architecture->is_waveform_encoder()))
        throw std::runtime_error("CLAP: this model does not expose frame representations");
    if (!(sample_rate > 0))
        throw std::runtime_error("CLAP: audio sample rate must be positive");
    const int target = model.architecture->sample_rate();
    if (sample_rate == target)
        return model.architecture->audio_frames(pcm);
    return model.architecture->audio_frames(resample_audio({pcm, sample_rate}, target).samples);
}
} // namespace clap::detail
