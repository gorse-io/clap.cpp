#pragma once

#include "clap.h"
#include "graph.h"
#include "model-loader.h"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace clap {
// Time-major waveform encoder output; values contains frames * dimension elements.
struct AudioFrames {
    std::vector<float> values;
    int frames = 0;
    int dimension = 0;
};
} // namespace clap

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
    virtual AudioFrames audio_frames(const std::vector<float> &pcm) const = 0;
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
    virtual void load_arch_hparams() = 0;
    virtual void load_arch_tensors() const = 0;
};
// CLAP2: NaFlex audio ViT and bidirectional ModernText, independent of HTSAT CLAP.
class Clap2Model final : public ModelArchitecture {
  public:
    Clap2Model(const ModelLoader &, int);
    std::vector<float> text(const std::vector<int> &) const override;
    std::vector<float> waveform_embedding(const std::vector<float> &, size_t = 0) const override;
    AudioFrames audio_frames(const std::vector<float> &) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override { return false; }
    bool has_waveform_embeddings() const override { return true; }

  private:
    friend struct FeatureTestAccess;
    void load_arch_hparams() override;
    void load_arch_tensors() const override;
    HostMatrix frontend(const std::vector<float> &) const;
    HostMatrix patches(const HostMatrix &) const;
    ggml_tensor *rms(Graph &, ggml_tensor *, const std::string &, float) const;
    ggml_tensor *rotate(Graph &, ggml_tensor *, int, float, int, int) const;
    ggml_tensor *attend(Graph &, ggml_tensor *, ggml_tensor *, ggml_tensor *, int, ggml_tensor *) const;
    ggml_tensor *encode(Graph &, ggml_tensor *, bool, int, ggml_tensor *) const;
    ggml_tensor *pool(Graph &, ggml_tensor *, bool, ggml_tensor *) const;
    const ModelLoader &loader;
    GraphRuntime runtime;
    nlohmann::json config, audio, txt, processor;
    int bins = 0, patch = 0, rate = 0, nfft = 0, hop = 0;
    float fmin = 0, fmax = 0;
};
class Vocabulary;
} // namespace clap::detail

// Definition of the public opaque C handle, not a second C++ API or facade.
// Reverse destruction order keeps the GGUF mapping alive for its architecture.
struct CLAP {
    clap::detail::ModelLoader loader;
    std::unique_ptr<clap::detail::ModelArchitecture> architecture;
    std::unique_ptr<clap::detail::Vocabulary> vocabulary;

    CLAP(const std::string &path, const clap::RuntimeOptions &options);
    ~CLAP();
};

namespace clap::detail {
// Core-only operations shared by the C entry points. Upper layers use clap.h.
std::vector<int> tokenize(const CLAP &model, const std::string &utf8, size_t max_length);
std::vector<float> encode_text(CLAP &model, const std::string &utf8);
std::vector<float> encode_audio(CLAP &model, const std::vector<float> &pcm, int sample_rate, size_t crop_offset);
std::vector<float> encode_audio_file(CLAP &model, const std::string &path, size_t crop_offset);
AudioFrames audio_frames(CLAP &model, const std::vector<float> &pcm, int sample_rate);
} // namespace clap::detail
