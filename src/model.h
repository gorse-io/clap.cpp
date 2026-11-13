#pragma once

#include "clap.h"
#include "graph.h"
#include "model-loader.h"

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
// Model-level loading orchestration; individual rules live in src/models.
class Context;
void load_model_weights(ModelLoader &, const std::string &, bool inference = true);
TensorLoadPolicy encoder_loading_policy(void (*validate_type)(const ggml_tensor *));
void validate_encoder_storage(const ggml_tensor *, bool quantizable);

// Shared model tensors. Creation validates each GGUF tensor and binds the
// returned pointer in one declaration; graphs never resolve tensor names.
LayerWeights create_layer(const ModelLoader &loader, const std::string &name,
                          std::initializer_list<int64_t> weight_dims, std::initializer_list<int64_t> bias_dims = {});
LayerWeights create_linear(const ModelLoader &loader, const std::string &name, int64_t input, int64_t output,
                           bool bias = true);
LayerWeights create_norm(const ModelLoader &loader, const std::string &name, int64_t width);
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
EmbeddingWeights create_embeddings(const ModelLoader &loader, const std::string &prefix, int64_t hidden, int64_t vocab,
                                   int64_t positions, int64_t types, bool word = true);
TransformerWeights create_bert_layer(const ModelLoader &loader, const std::string &prefix, int64_t hidden,
                                     int64_t intermediate);
struct BatchNormWeights {
    LayerWeights affine;
    ggml_tensor *mean = nullptr, *variance = nullptr;
    static BatchNormWeights create(const ModelLoader &loader, const std::string &prefix, int64_t width);
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
    virtual void initialize_context(Context &) const = 0;
    virtual std::vector<float> text(Context &, const std::vector<int> &ids) const = 0;
    virtual AudioFrames audio_frames(Context &, const std::vector<float> &pcm) const = 0;
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
    virtual std::vector<float> waveform_embedding(Context &, const std::vector<float> &, size_t = 0) const {
        throw std::runtime_error("CLAP: this architecture has no waveform embeddings");
    }

  protected:
    virtual void load_arch_hparams() = 0;
    virtual void load_arch_tensors() = 0;
};
class Vocabulary;
struct ModelWeightStats {
    size_t host_tensors = 0, device_tensors = 0;
    size_t host_bytes = 0, device_bytes = 0, uploads = 0;
};

// Model-lifetime, read-only leaf tensors and allocation backends. A completed
// entry is shared by device/buffer/layout identity, never by a compute backend.
class ModelWeights {
  public:
    explicit ModelWeights(const ModelLoader &);
    ~ModelWeights();
    ggml_backend_dev_t gpu_device() const;
    bool uses_vulkan() const;
    bool uses_metal() const;
    int default_threads() const;
    ModelWeightStats stats() const;
    ggml_tensor *get(ggml_tensor *source, ggml_backend_t compute, bool device, bool &uploaded);

  private:
    struct State;
    std::unique_ptr<State> state;
};

// Configuration, vocabulary and source tensors are immutable after construction.
// Contexts retain this owner; destruction frees buffers before the GGUF mapping.
struct Model {
    ModelLoader loader;
    std::shared_ptr<ModelWeights> weights;
    std::unique_ptr<const ModelArchitecture> architecture;
    std::unique_ptr<const Vocabulary> vocabulary;

    Model(const std::string &path, const RuntimeOptions &options);
    ~Model();
};
} // namespace clap::detail

struct clap_model {
    std::shared_ptr<const clap::detail::Model> value;
};
