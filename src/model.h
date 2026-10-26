#pragma once

#include "clap.h"
#include <cstddef>
#include <memory>
#include <vector>
#include <stdexcept>

namespace clap::detail {
class ModelContext;
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
    virtual AudioFrames audio_frames(const std::vector<float> &pcm) const = 0;
    virtual AudioFrames clamp3_features(const std::vector<float> &) const {
        throw std::runtime_error("CLaMP3 feature extraction requires a MERT-v1 model");
    }
    virtual TextParameters text_parameters() const = 0;
    virtual int projection_dimension() const = 0;
    virtual int mel_bins() const = 0;
    virtual int sample_rate() const = 0;
    // Historical name: this capability exposes waveform frame sequences.
    // Joint models with their own clip pooling use has_waveform_embeddings().
    virtual bool is_waveform_encoder() const = 0;
    // A joint waveform model has a tokenizer and its own clip/embedding pooling.
    virtual bool has_text_encoder() const { return !is_waveform_encoder(); }
    virtual bool has_waveform_embeddings() const { return false; }
    virtual std::vector<float> waveform_embedding(const std::vector<float> &, size_t = 0) const {
        throw std::runtime_error("CLAP: this architecture has no waveform embeddings");
    }
protected:
    // Architecture-local execution is constructed only after parameters and
    // tensor contracts are validated, like llama.cpp build_arch_graph().
    virtual std::unique_ptr<ModelContext> build_arch_context(int threads) const = 0;
    virtual void load_arch_hparams() = 0;
    virtual void load_arch_tensors() const = 0;
};
} // namespace clap::detail
