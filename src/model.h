#pragma once

#include "clap.h"
#include "graph.h"
#include "model-loader.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace clap {
// Time-major waveform encoder output; values contains frames * dimension elements.
struct AudioFrames {
    std::vector<float> values;
    int frames = 0;
    int dimension = 0;
};
} // namespace clap

namespace clap::detail {
// Shared model tensors, bound once after GGUF shape validation. Architecture
// graphs consume these pointers without resolving tensor names.
struct AttentionWeights {
    LayerWeights query, key, value, output;
    ggml_tensor *relative_bias = nullptr;
};
struct TransformerWeights {
    AttentionWeights attention;
    LayerWeights attention_norm, output_norm, intermediate, output;
};
struct EmbeddingWeights {
    ggml_tensor *word = nullptr, *position = nullptr, *type = nullptr;
    LayerWeights norm;
};
EmbeddingWeights bind_embeddings(const ModelLoader &loader, const std::string &prefix, bool word = true);
TransformerWeights bind_bert_layer(const ModelLoader &loader, const std::string &prefix);
struct BatchNormWeights {
    LayerWeights affine;
    ggml_tensor *mean = nullptr, *variance = nullptr;
    static BatchNormWeights bind(const ModelLoader &loader, const std::string &prefix);
};
struct FeedForwardWeights {
    LayerWeights norm, input, output;
};
struct ConvolutionWeights {
    LayerWeights norm, input, depthwise, output_norm, output;
    BatchNormWeights batch_norm;
};
struct ConformerWeights {
    FeedForwardWeights ffn1, ffn2;
    LayerWeights attention_norm, final_norm;
    AttentionWeights attention;
    ConvolutionWeights convolution;
};

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
