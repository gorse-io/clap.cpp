#include "audio.h"
#include "model.h"
#include "models/models.h"
#include "quant.h"
#include "vocab.h"

#include <nlohmann/json.hpp>

#include <stdexcept>

CLAP::CLAP(const std::string &path, const clap::RuntimeOptions &options) : loader(options) {
    using namespace clap::detail;
    const int threads = options.threads;
    clap::detail::load_model_weights(loader, path);
    const auto name = loader.metadata("general.architecture");
    if (name == "clap")
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
    else
        throw std::runtime_error("CLAP: unsupported GGUF architecture: " + name);
    loader.validate_tensor_coverage();
    if (architecture->has_text_encoder()) {
        const auto text = architecture->text_parameters();
        const auto tokenizer = nlohmann::json::parse(loader.metadata((name + ".tokenizer").c_str()));
        if (name == "clamp3" || name == "muq_mulan")
            vocabulary = std::make_unique<UnigramVocabulary>(tokenizer, text.bos, text.eos, 3, text.context_length);
        else if (name == "clap2")
            vocabulary = std::make_unique<ByteBpeVocabulary>(tokenizer, text.bos, text.eos, text.context_length, true);
        else
            vocabulary = std::make_unique<ByteBpeVocabulary>(tokenizer, text.bos, text.eos, text.context_length);
    }
}
CLAP::~CLAP() = default;

namespace clap::detail {
namespace {
TensorLoadPolicy select_loading_policy(const ModelLoader &loader) {
    std::string architecture;
    try {
        architecture = loader.metadata("general.architecture");
    } catch (const std::runtime_error &) {
        // Missing or non-string architecture metadata is neutral at this layer.
        // The model facade retains ownership of full architecture validation.
        return {};
    }
    if (architecture == "clap")
        return ClapModel::loading_policy(loader);
    if (architecture == "clap2")
        return Clap2Model::loading_policy(loader);
    if (architecture == "clamp3")
        return Clamp3Model::loading_policy(loader);
    if (architecture == "mert")
        return MertModel::loading_policy(loader);
    if (architecture == "mert2")
        return Mert2Model::loading_policy(loader);
    if (architecture == "muq")
        return MuqModel::loading_policy(loader);
    if (architecture == "muq_mulan")
        return MuqMulanModel::loading_policy(loader);
    return {};
}
} // namespace

void load_model_weights(ModelLoader &loader, const std::string &path, bool inference) {
    loader.load(path, inference, select_loading_policy);
}

TensorLoadPolicy encoder_loading_policy(void (*validate_type)(const ggml_tensor *)) {
    TensorLoadPolicy policy;
    policy.f32_activations = true;
    policy.unrecognized_tensor_error = "CLAP: unrecognized model tensor: ";
    policy.validate_tensor_type = validate_type;
    return policy;
}

void validate_encoder_storage(const ggml_tensor *w, bool quantizable) {
    if (ggml_is_quantized(w->type) && !quantizable)
        throw std::runtime_error("CLAP: quantized non-dense/embedding tensor is unsupported: " + std::string(w->name));
}

bool encoder_quantizable(const ggml_tensor *w, const std::string &architecture) {
    if (architecture == "mert")
        return MertModel::tensor_quantizable(w);
    if (architecture == "mert2")
        return Mert2Model::tensor_quantizable(w);
    if (architecture == "muq")
        return MuqModel::tensor_quantizable(w);
    if (architecture == "muq_mulan")
        return MuqMulanModel::tensor_quantizable(w);
    return false;
}

LayerWeights create_layer(const ModelLoader &loader, const std::string &name,
                          std::initializer_list<int64_t> weight_dims, std::initializer_list<int64_t> bias_dims) {
    auto *weight = loader.create_tensor(name + ".weight", weight_dims);
    auto *bias = bias_dims.size() ? loader.create_tensor(name + ".bias", bias_dims) : nullptr;
    return {weight, bias};
}
LayerWeights create_linear(const ModelLoader &loader, const std::string &name, int64_t input, int64_t output,
                           bool bias) {
    return bias ? create_layer(loader, name, {input, output}, {output}) : create_layer(loader, name, {input, output});
}
LayerWeights create_norm(const ModelLoader &loader, const std::string &name, int64_t width) {
    return create_layer(loader, name, {width}, {width});
}
EmbeddingWeights create_embeddings(const ModelLoader &loader, const std::string &prefix, int64_t hidden, int64_t vocab,
                                   int64_t positions, int64_t types, bool word) {
    return {word ? loader.create_tensor(prefix + ".word_embeddings.weight", {hidden, vocab}) : nullptr,
            loader.create_tensor(prefix + ".position_embeddings.weight", {hidden, positions}),
            loader.create_tensor(prefix + ".token_type_embeddings.weight", {hidden, types}),
            create_norm(loader, prefix + ".LayerNorm", hidden)};
}
TransformerWeights create_bert_layer(const ModelLoader &loader, const std::string &prefix, int64_t hidden,
                                     int64_t intermediate) {
    TransformerWeights layer;
    layer.attention = {create_linear(loader, prefix + ".attention.self.query", hidden, hidden),
                       create_linear(loader, prefix + ".attention.self.key", hidden, hidden),
                       create_linear(loader, prefix + ".attention.self.value", hidden, hidden),
                       create_linear(loader, prefix + ".attention.output.dense", hidden, hidden)};
    layer.attention_norm = create_norm(loader, prefix + ".attention.output.LayerNorm", hidden);
    layer.intermediate = create_linear(loader, prefix + ".intermediate.dense", hidden, intermediate);
    layer.output = create_linear(loader, prefix + ".output.dense", intermediate, hidden);
    layer.output_norm = create_norm(loader, prefix + ".output.LayerNorm", hidden);
    return layer;
}
BatchNormWeights BatchNormWeights::create(const ModelLoader &loader, const std::string &prefix, int64_t width) {
    return {create_norm(loader, prefix, width), loader.create_tensor(prefix + ".running_mean", {width}),
            loader.create_tensor(prefix + ".running_var", {width})};
}

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
