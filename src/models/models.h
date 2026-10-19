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
} // namespace clap::detail
