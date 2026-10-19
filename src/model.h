#pragma once

#include <cstddef>
#include <vector>

namespace clap::detail {
struct TextParameters {
    int bos;
    int eos;
    size_t context_length;
};

// Internal architecture contract, analogous to llama.cpp's model specializations.
// Public API, shared loading, tokenization and audio preprocessing live outside
// src/models; each architecture supplies its parameter/tensor contract and encoders.
class ModelArchitecture {
public:
    virtual ~ModelArchitecture() = default;
    virtual std::vector<float> text(const std::vector<int> &ids) const = 0;
    virtual std::vector<float> audio_features(const std::vector<float> &features, int frames) const = 0;
    virtual TextParameters text_parameters() const = 0;
    virtual int projection_dimension() const = 0;
    virtual int mel_bins() const = 0;
protected:
    virtual void load_arch_hparams() = 0;
    virtual void load_arch_tensors() const = 0;
};
} // namespace clap::detail
