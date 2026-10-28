#pragma once

#include "model.h"
#include "model-loader.h"
#include "model-ops.h"
#include <nlohmann/json.hpp>

namespace clap::detail {

// Per-architecture declarations belong here, following llama.cpp/src/models.
// GGUF I/O, tokenizer ownership, generic tensor operations and the public CLAP
// facade are implemented in the core src/ files, not in this specialization.
class ClapModel final : public ModelArchitecture {
public:
    ClapModel(const ModelLoader &loader, int threads);
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
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    Matrix attention(const Matrix &x, const std::string &prefix, int heads, int tokens,
                     const std::vector<int> *mask = nullptr, int window = 0, int shift = 0, int grid = 0) const;
    std::vector<float> project(const Matrix &pooled, const std::string &prefix) const;

    const ModelLoader &loader;
    TensorOps ops;
    nlohmann::json config, a, t;
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
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    Matrix encode(const Matrix &embeddings, const std::vector<int> &mask,
                  const nlohmann::json &branch, const std::string &prefix) const;
    Matrix attention(const Matrix &x, const std::vector<int> &mask,
                     const std::string &prefix, int heads) const;
    std::vector<float> chunked_text(const std::vector<int> &ids) const;
    std::vector<float> chunked_audio(const Matrix &features) const;
    std::vector<float> pool_project(const Matrix &hidden, const std::vector<int> &mask,
                                    const std::string &projection) const;

    const ModelLoader &loader;
    TensorOps ops;
    nlohmann::json config, audio_config, text_config;
    std::unique_ptr<MertModel> frontend;
};

class GemmaAudioEncoder;
// Gemma-embedding2 GGUF shared text/soft-token transformer.
class EmbeddingGemma2Model final : public ModelArchitecture {
public:
    EmbeddingGemma2Model(const ModelLoader &loader, int threads, const ModelLoader *audio_loader = nullptr);
    ~EmbeddingGemma2Model() override;
    std::vector<float> text(const std::vector<int> &ids) const override;
    std::vector<float> waveform_embedding(const std::vector<float> &pcm, size_t crop_offset = 0) const override;
    bool has_waveform_embeddings() const override { return true; }
    AudioFrames audio_frames(const std::vector<float> &) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override { return output; }
    int mel_bins() const override { return 0; }
    int sample_rate() const override { return 16000; }
    bool is_waveform_encoder() const override { return false; }
private:
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    Matrix rms(const Matrix &x, const std::string &name, int width = 0) const;
    Matrix attention(const Matrix &x, int layer, const std::vector<int> &mask) const;
    friend struct FeatureTestAccess;
    std::vector<float> encode(Matrix x, const std::vector<int> &mask) const;
    const ModelLoader &loader;
    TensorOps ops;
    std::unique_ptr<GemmaAudioEncoder> audio_encoder;
    int begin_audio = -1, end_audio = -1;
    int hidden = 0, output = 0, layers = 0, per_layer = 0, vocab = 0, context = 0, window = 0;
    int bos = 2, eos = 1, pad = 0;
    float epsilon = 0, rope_base = 0, rope_base_swa = 0, rope_scale = 1;
    std::vector<int> heads, kv_heads, dimensions, rotations, ffn;
    std::vector<bool> local;
};

// Official m-a-p/MERT-v2-FullSong waveform encoder: checkpoint frontend,
// ConvNeXt subsampler and bidirectional rotary Conformer stack.
class Mert2Model final : public ModelArchitecture {
public:
    Mert2Model(const ModelLoader &loader, int threads);
    std::vector<float> text(const std::vector<int> &ids) const override;
    AudioFrames audio_frames(const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;
private:
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    Matrix frontend(const std::vector<float> &pcm) const;
    Matrix subsample(Matrix hidden) const;
    Matrix attention(const Matrix &hidden, const std::string &prefix) const;
    Matrix conformer_convolution(const Matrix &hidden, const std::string &prefix) const;

    const ModelLoader &loader;
    TensorOps ops;
    nlohmann::json config;
};
// MERT-v1-95M: normalized waveform, HuBERT convolutional frontend and
// post-layer-normalized transformer. CLaMP3 pooling is a separate contract.
class MertModel final : public ModelArchitecture {
public:
    MertModel(const ModelLoader &loader, int threads,
              const std::string &tensor_prefix = "", const std::string &metadata_prefix = "");
    std::vector<float> text(const std::vector<int> &) const override;
    AudioFrames audio_frames(const std::vector<float> &pcm) const override;
    AudioFrames clamp3_features(const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override { return {0,0,0}; }
    int projection_dimension() const override { return config.at("hidden_size"); }
    int mel_bins() const override { return 0; }
    int sample_rate() const override { return 24000; }
    bool is_waveform_encoder() const override { return true; }
private:
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    std::vector<float> normalize(const std::vector<float> &pcm) const;
    Matrix forward(const std::vector<float> &normalized, std::vector<float> *layer_mean) const;
    Matrix convolution(const Matrix &x, const std::string &name, int kernel,
                       int stride, int padding, int groups, bool bias) const;
    Matrix linear(const Matrix &x, const std::string &name) const;
    Matrix attention(const Matrix &x, const std::string &name) const;
    ModelLoader loader;
    TensorOps ops;
    nlohmann::json config;
    int minimum_samples = 0;
};

// OpenMuQ/MuQ-large-msd-iter: native log-mel, residual 2D subsampling and
// Wav2Vec2-Conformer (rotary BEFORE Q/K projections, SiLU and eval BatchNorm).
class MuqModel final : public ModelArchitecture {
public:
    MuqModel(const ModelLoader &loader, int threads);
    std::vector<float> text(const std::vector<int> &) const override;
    AudioFrames audio_frames(const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;
private:
    struct Image { int frequencies; int frames; Matrix values; };
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    Matrix frontend(const std::vector<float> &pcm) const;
    Matrix subsample(const Matrix &mel) const;
    Image conv2d(const Image &input, const std::string &prefix, int stride) const;
    void batch_norm(Matrix &input, const std::string &prefix) const;
    Matrix attention(const Matrix &hidden, const std::string &prefix) const;
    Matrix conformer_convolution(const Matrix &hidden, const std::string &prefix) const;
    const ModelLoader &loader;
    TensorOps ops;
    nlohmann::json config, encoder;
};

// Joint MuQ-MuLan embeddings: the audio backbone remains the exact MuqModel.
class MuqMulanModel final : public ModelArchitecture {
public:
    MuqMulanModel(const ModelLoader &loader, int threads);
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
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    Matrix encode_text(const Matrix &x) const;
    Matrix adapter(Matrix x) const;
    Matrix biasless_norm(const Matrix &x, const std::string &name) const;
    std::vector<float> project(const Matrix &x, const std::string &name) const;
    const ModelLoader &loader;
    TensorOps ops;
    MuqModel audio_backbone;
    nlohmann::json config, text_config;
};
} // namespace clap::detail
