#pragma once

#include "model.h"
#include "model-loader.h"
#include <nlohmann/json.hpp>

namespace clap::detail {

// Per-architecture declarations belong here, following llama.cpp/src/models.
// Model specializations own validated configuration, tensor contracts and embedded
// backbones. Their nested contexts (defined in each .cpp) own inference helpers
// and execution state. GGUF I/O, vocabulary and the Model facade stay in core.
class ClapModel final : public ModelArchitecture {
public:
    ClapModel(const ModelLoader &loader, int threads);
    ~ClapModel() override;
    std::vector<float> text(const std::vector<int> &ids) const override;
    AudioFrames audio_frames(const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;
    std::vector<float> waveform_embedding(const std::vector<float> &pcm, size_t crop_offset = 0) const override;
    bool has_waveform_embeddings() const override { return true; }
private:
    friend struct FeatureTestAccess;
    std::vector<float> encode_features(const std::vector<float> &features, int frames) const;
    struct context;
    std::unique_ptr<ModelContext> build_arch_context(int threads) const override;
    void load_arch_hparams() override;
    void load_arch_tensors() const override;

    const ModelLoader &loader;
    nlohmann::json config, a, t;
    std::unique_ptr<ModelContext> inference;
};

// CLaMP3 text (XLM-R) and audio (BERT), optionally with a bundled MERT frontend.
class MertModel;
class Clamp3Model final : public ModelArchitecture {
public:
    Clamp3Model(const ModelLoader &loader, int threads);
    ~Clamp3Model() override;
    bool has_waveform_embeddings() const override;
    std::vector<float> waveform_embedding(const std::vector<float> &pcm, size_t crop_offset = 0) const override;
    std::vector<float> text(const std::vector<int> &ids) const override;
    AudioFrames audio_frames(const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;
private:
    friend struct FeatureTestAccess;
    std::vector<float> encode_features(const std::vector<float> &features, int frames) const;
    struct context;
    std::unique_ptr<ModelContext> build_arch_context(int threads) const override;
    void load_arch_hparams() override;
    void load_arch_tensors() const override;

    const ModelLoader &loader;
    nlohmann::json config, audio_config, text_config;
    std::unique_ptr<MertModel> frontend;
    std::unique_ptr<ModelContext> inference;
};

// Official m-a-p/MERT-v2-FullSong waveform encoder: checkpoint frontend,
// ConvNeXt subsampler and bidirectional rotary Conformer stack.
class Mert2Model final : public ModelArchitecture {
public:
    Mert2Model(const ModelLoader &loader, int threads);
    ~Mert2Model() override;
    std::vector<float> text(const std::vector<int> &ids) const override;
    AudioFrames audio_frames(const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;
private:
    struct context;
    std::unique_ptr<ModelContext> build_arch_context(int threads) const override;
    void load_arch_hparams() override;
    void load_arch_tensors() const override;

    const ModelLoader &loader;
    nlohmann::json config;
    std::unique_ptr<ModelContext> inference;
};
// MERT-v1-95M: normalized waveform, HuBERT convolutional frontend and
// post-layer-normalized transformer. CLaMP3 pooling is a separate contract.
class MertModel final : public ModelArchitecture {
public:
    MertModel(const ModelLoader &loader, int threads,
              const std::string &tensor_prefix = "", const std::string &metadata_prefix = "");
    ~MertModel() override;
    std::vector<float> text(const std::vector<int> &) const override;
    AudioFrames audio_frames(const std::vector<float> &pcm) const override;
    AudioFrames clamp3_features(const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override { return {0,0,0}; }
    int projection_dimension() const override { return config.at("hidden_size"); }
    int mel_bins() const override { return 0; }
    int sample_rate() const override { return 24000; }
    bool is_waveform_encoder() const override { return true; }
private:
    struct context;
    std::unique_ptr<ModelContext> build_arch_context(int threads) const override;
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    ModelLoader loader;
    nlohmann::json config;
    int minimum_samples = 0;
    std::unique_ptr<ModelContext> inference;
};

// OpenMuQ/MuQ-large-msd-iter: native log-mel, residual 2D subsampling and
// Wav2Vec2-Conformer (rotary BEFORE Q/K projections, SiLU and eval BatchNorm).
class MuqModel final : public ModelArchitecture {
public:
    MuqModel(const ModelLoader &loader, int threads);
    ~MuqModel() override;
    std::vector<float> text(const std::vector<int> &) const override;
    AudioFrames audio_frames(const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;
private:
    struct context;
    std::unique_ptr<ModelContext> build_arch_context(int threads) const override;
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    const ModelLoader &loader;
    nlohmann::json config, encoder;
    std::unique_ptr<ModelContext> inference;
};

// Joint MuQ-MuLan embeddings: the audio backbone remains the exact MuqModel.
class MuqMulanModel final : public ModelArchitecture {
public:
    MuqMulanModel(const ModelLoader &loader, int threads);
    ~MuqMulanModel() override;
    std::vector<float> text(const std::vector<int> &ids) const override;
    AudioFrames audio_frames(const std::vector<float> &) const override;
    std::vector<float> waveform_embedding(const std::vector<float> &pcm, size_t crop_offset = 0) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override { return 0; }
    int sample_rate() const override { return 24000; }
    bool is_waveform_encoder() const override { return false; }
    bool has_waveform_embeddings() const override { return true; }
private:
    struct context;
    std::unique_ptr<ModelContext> build_arch_context(int threads) const override;
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    const ModelLoader &loader;
    MuqModel audio_backbone;
    nlohmann::json config, text_config;
    std::unique_ptr<ModelContext> inference;
};
} // namespace clap::detail
