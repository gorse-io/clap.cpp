#pragma once

#include "model.h"
#include "model-loader.h"
#include "model-ops.h"
#include <nlohmann/json.hpp>

namespace clap::detail {

// Per-architecture declarations belong here, following llama.cpp/src/models.
// GGUF I/O, tokenizer ownership, generic tensor operations and the public Model
// facade are implemented in the core src/ files, not in this specialization.
class ClapModel final : public ModelArchitecture {
public:
    ClapModel(const ModelLoader &loader, int threads);
    std::vector<float> text(const std::vector<int> &ids) const override;
    std::vector<float> audio_features(const std::vector<float> &features, int frames) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
private:
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    Matrix attention(const Matrix &x, const std::string &prefix, int heads, int tokens,
                     const std::vector<int> *mask = nullptr, int window = 0, int shift = 0, int grid = 0) const;
    std::vector<float> project(const Matrix &pooled, const std::string &prefix) const;

    const ModelLoader &loader;
    TensorOps ops;
    nlohmann::json config, a, t;
};

// CLaMP3 text (XLM-R) and precomputed-audio (BERT) encoders. Raw MERT feature
// extraction and the optional symbolic encoder intentionally remain external.
class Clamp3Model final : public ModelArchitecture {
public:
    Clamp3Model(const ModelLoader &loader, int threads);
    std::vector<float> text(const std::vector<int> &ids) const override;
    std::vector<float> audio_features(const std::vector<float> &features, int frames) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
private:
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
};
} // namespace clap::detail
