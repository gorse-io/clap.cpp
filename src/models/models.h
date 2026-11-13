/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "graph.h"
#include "model-loader.h"
#include "model.h"

#include <nlohmann/json.hpp>

namespace clap::detail {

// Per-architecture declarations belong here, following llama.cpp/src/models.
// GGUF I/O, tokenizer ownership, generic tensor operations and the public CLAP
// facade are implemented in the core src/ files, not in this specialization.
class ClapModel final : public ModelArchitecture {
  public:
    static TensorLoadPolicy loading_policy(const ModelLoader &);
    void initialize_context(Context &) const override;
    ClapModel(const ModelLoader &loader);
    std::vector<float> text(Context &, const std::vector<int> &ids) const override;
    AudioFrames audio_frames(Context &, const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;
    std::vector<float> waveform_embedding(Context &, const std::vector<float> &pcm,
                                          size_t crop_offset = 0) const override;
    bool has_waveform_embeddings() const override { return true; }

  private:
    friend struct FeatureTestAccess;
    std::vector<float> encode_features(Context &, const std::vector<float> &features, int frames) const;
    void load_arch_hparams() override;
    void load_arch_tensors() override;
    ggml_tensor *attention(Graph &g, ggml_tensor *x, const AttentionWeights &weights, int heads, int tokens,
                           ggml_tensor *mask = nullptr, int window = 0, int shift = 0, int grid = 0) const;
    struct Projection;
    ggml_tensor *project(Graph &g, ggml_tensor *pooled, const Projection &weights) const;

    const ModelLoader &loader;
    nlohmann::json config, a, t;
    struct Parameters {
        int projection, bins, spec, patch_width, window, hidden, pad, vocab, positions, heads, bos, eos;
        float audio_epsilon, text_epsilon;
    } h;
    struct Stage {
        int heads;
        std::vector<TransformerWeights> blocks;
        LayerWeights norm, reduction;
    };
    std::vector<Stage> stages;
    EmbeddingWeights embeddings;
    std::vector<TransformerWeights> text_layers;
    LayerWeights patch, patch_norm, audio_norm, pooler;
    struct Projection {
        LayerWeights input, output;
    } audio_projection, text_projection;
    BatchNormWeights batch_norm;
};

// CLAP2: NaFlex audio ViT and bidirectional ModernText, independent of HTSAT CLAP.
class Clap2Model final : public ModelArchitecture {
  public:
    static TensorLoadPolicy loading_policy(const ModelLoader &);
    void initialize_context(Context &) const override;
    Clap2Model(const ModelLoader &);
    std::vector<float> text(Context &, const std::vector<int> &) const override;
    std::vector<float> waveform_embedding(Context &, const std::vector<float> &, size_t = 0) const override;
    AudioFrames audio_frames(Context &, const std::vector<float> &) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override { return false; }
    bool has_waveform_embeddings() const override { return true; }

  private:
    friend struct FeatureTestAccess;
    void load_arch_hparams() override;
    void load_arch_tensors() override;
    HostMatrix frontend(const std::vector<float> &) const;
    HostMatrix patches(const HostMatrix &) const;
    ggml_tensor *rms(Graph &, ggml_tensor *, ggml_tensor *, float) const;
    ggml_tensor *rotate(Graph &, ggml_tensor *, int, float, int, int) const;
    ggml_tensor *attend(Graph &, ggml_tensor *, ggml_tensor *, ggml_tensor *, int, ggml_tensor *) const;
    ggml_tensor *encode(Graph &, ggml_tensor *, bool, int, ggml_tensor *) const;
    ggml_tensor *pool(Graph &, ggml_tensor *, bool, ggml_tensor *) const;
    const ModelLoader &loader;
    struct Parameters {
        int projection, bins, patch, rate, nfft, hop;
        int vocab, positions, bos, eos, pad;
        float fmin, fmax;
    } h;
    struct BranchParameters {
        int hidden, heads, intermediate, layers;
        float epsilon, theta;
    };
    struct Layer {
        ggml_tensor *input_norm = nullptr, *post_attention_norm = nullptr;
        ggml_tensor *q_norm = nullptr, *k_norm = nullptr;
        ggml_tensor *attention_scale = nullptr, *mlp_scale = nullptr;
        LayerWeights qkv, attention_output, gate_up, down;
    };
    struct Head {
        ggml_tensor *probe = nullptr, *norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        LayerWeights query, key_value, output, fc1, fc2;
    };
    struct Branch {
        BranchParameters h;
        std::vector<Layer> layers;
        ggml_tensor *norm = nullptr;
        Head head;
    } audio_branch, text_branch;
    ggml_tensor *text_embeddings = nullptr, *register_tokens = nullptr, *patch_norm = nullptr;
    LayerWeights patch_embedding, audio_projection_input, audio_projection_output, text_projection;
};

// CLaMP3 text (XLM-R) and audio (BERT), optionally with a bundled MERT frontend.
class MertModel;
class Clamp3Model final : public ModelArchitecture {
  public:
    static TensorLoadPolicy loading_policy(const ModelLoader &);
    void initialize_context(Context &) const override;
    Clamp3Model(const ModelLoader &loader);
    ~Clamp3Model() override;
    bool has_waveform_embeddings() const override;
    std::vector<float> waveform_embedding(Context &, const std::vector<float> &pcm,
                                          size_t crop_offset = 0) const override;
    std::vector<float> text(Context &, const std::vector<int> &ids) const override;
    AudioFrames audio_frames(Context &, const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;

  private:
    friend struct FeatureTestAccess;
    std::vector<float> encode_features(Context &, const std::vector<float> &features, int frames) const;
    void load_arch_hparams() override;
    void load_arch_tensors() override;
    struct Branch;
    ggml_tensor *encode(Graph &g, ggml_tensor *embeddings, const std::vector<int> &mask, const Branch &branch) const;
    ggml_tensor *attention(Graph &g, ggml_tensor *x, const std::vector<int> &mask, const AttentionWeights &weights,
                           int heads) const;
    std::vector<float> chunked_text(Context &, const std::vector<int> &ids) const;
    std::vector<float> chunked_audio(Context &, const HostMatrix &features) const;
    ggml_tensor *pool_project(Graph &g, ggml_tensor *hidden, const std::vector<int> &mask,
                              const LayerWeights &projection) const;

    const ModelLoader &loader;
    nlohmann::json config, audio_config, text_config;
    std::unique_ptr<MertModel> frontend;
    struct Branch {
        int hidden, heads, pad, vocab, positions, bos, eos;
        float epsilon;
        EmbeddingWeights embeddings;
        std::vector<TransformerWeights> layers;
        LayerWeights projection;
    };
    Branch audio_branch, text_branch;
    int chunk_length = 128, projection_dim = 0;
};

// Official m-a-p/MERT-v2-FullSong waveform encoder: checkpoint frontend,
// ConvNeXt subsampler and bidirectional rotary Conformer stack.
class Mert2Model final : public ModelArchitecture {
  public:
    static TensorLoadPolicy loading_policy(const ModelLoader &);
    void initialize_context(Context &) const override;
    static bool tensor_quantizable(const ggml_tensor *);
    Mert2Model(const ModelLoader &loader);
    std::vector<float> text(Context &, const std::vector<int> &ids) const override;
    AudioFrames audio_frames(Context &, const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;

  private:
    void load_arch_hparams() override;
    void load_arch_tensors() override;
    HostMatrix frontend(const std::vector<float> &pcm) const;
    ggml_tensor *subsample(Graph &g, ggml_tensor *hidden) const;
    ggml_tensor *attention(Graph &g, ggml_tensor *hidden, const AttentionWeights &weights) const;
    ggml_tensor *conformer_convolution(Graph &g, ggml_tensor *hidden, const ConvolutionWeights &weights) const;

    const ModelLoader &loader;
    nlohmann::json config;
    struct Parameters {
        int hidden, bins, sample_rate, fft, hop, heads, kernel;
        size_t minimum_samples;
        float epsilon, subsampling_epsilon, rotary_base;
    } h;
    ggml_tensor *window = nullptr, *filters = nullptr, *mean = nullptr, *stddev = nullptr;
    struct ConvNext {
        LayerWeights depthwise, norm, input, response, output;
    };
    struct Stage {
        int width;
        LayerWeights norm, resample;
        std::vector<ConvNext> blocks;
    };
    std::vector<Stage> stages;
    std::vector<ConformerWeights> layers;
};
// MERT-v1: normalized waveform, HuBERT convolutional frontend and pre/post-
// layer-normalized transformer. CLaMP3 pooling is a separate contract.
class MertModel final : public ModelArchitecture {
  public:
    static TensorLoadPolicy loading_policy(const ModelLoader &);
    void initialize_context(Context &) const override;
    static bool tensor_quantizable(const ggml_tensor *);
    MertModel(const ModelLoader &loader, const std::string &tensor_prefix = "",
              const std::string &metadata_prefix = "");
    std::vector<float> text(Context &, const std::vector<int> &) const override;
    AudioFrames audio_frames(Context &, const std::vector<float> &pcm) const override;
    // Internal CLaMP3 frontend: normalize the whole recording, split into 5 s
    // windows, discard a tail shorter than 1 s, and mean-pool time/all layers.
    AudioFrames clamp3_features(Context &, const std::vector<float> &pcm) const;
    TextParameters text_parameters() const override { return {0, 0, 0}; }
    int projection_dimension() const override { return h.hidden; }
    int mel_bins() const override { return 0; }
    int sample_rate() const override { return 24000; }
    bool is_waveform_encoder() const override { return true; }

  private:
    void load_arch_hparams() override;
    void load_arch_tensors() override;
    std::vector<float> normalize(const std::vector<float> &pcm) const;
    HostMatrix forward(Context &, const std::vector<float> &normalized, std::vector<float> *layer_mean) const;
    ggml_tensor *attention(Graph &g, ggml_tensor *x, const AttentionWeights &weights) const;
    ModelLoader loader;
    nlohmann::json config;
    int minimum_samples = 0;
    struct Parameters {
        int hidden, heads, position_kernel, position_groups;
        float epsilon;
        std::vector<int> channels, kernels, strides;
        bool conv_bias, stable_layer_norm;
    } h;
    std::vector<LayerWeights> convolutions;
    LayerWeights frontend_norm, projection_norm, projection, position, encoder_norm;
    std::vector<TransformerWeights> layers;
};

// OpenMuQ/MuQ-large-msd-iter: native log-mel, residual 2D subsampling and
// Wav2Vec2-Conformer (rotary BEFORE Q/K projections, SiLU and eval BatchNorm).
class MuqModel final : public ModelArchitecture {
  public:
    static TensorLoadPolicy loading_policy(const ModelLoader &);
    void initialize_context(Context &) const override;
    static bool tensor_quantizable(const ggml_tensor *);
    MuqModel(const ModelLoader &loader);
    std::vector<float> text(Context &, const std::vector<int> &) const override;
    AudioFrames audio_frames(Context &, const std::vector<float> &pcm) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override;
    int sample_rate() const override;
    bool is_waveform_encoder() const override;

  private:
    struct Image {
        int frequencies;
        int frames;
        ggml_tensor *values;
    };
    void load_arch_hparams() override;
    void load_arch_tensors() override;
    HostMatrix frontend(const std::vector<float> &pcm) const;
    ggml_tensor *subsample(Graph &g, ggml_tensor *mel) const;
    Image conv2d(Graph &g, const Image &input, const LayerWeights &weights, int stride) const;
    ggml_tensor *batch_norm(Graph &g, ggml_tensor *input, const BatchNormWeights &weights) const;
    ggml_tensor *attention(Graph &g, ggml_tensor *hidden, const AttentionWeights &weights) const;
    ggml_tensor *conformer_convolution(Graph &g, ggml_tensor *hidden, const ConvolutionWeights &weights) const;
    const ModelLoader &loader;
    nlohmann::json config, encoder;
    struct Parameters {
        int hidden, heads, kernel;
        float epsilon, rotary_base, mean, stddev;
    } h;
    ggml_tensor *window = nullptr, *filters = nullptr;
    struct Stage {
        LayerWeights conv1, conv2, conv3;
        BatchNormWeights bn1, bn2, bn3;
    };
    std::vector<Stage> stages;
    std::vector<ConformerWeights> layers;
    LayerWeights projection, final_norm;
};

// Joint MuQ-MuLan embeddings: the audio backbone remains the exact MuqModel.
class MuqMulanModel final : public ModelArchitecture {
  public:
    static TensorLoadPolicy loading_policy(const ModelLoader &);
    void initialize_context(Context &) const override;
    static bool tensor_quantizable(const ggml_tensor *);
    MuqMulanModel(const ModelLoader &loader);
    std::vector<float> text(Context &, const std::vector<int> &ids) const override;
    AudioFrames audio_frames(Context &, const std::vector<float> &) const override;
    std::vector<float> waveform_embedding(Context &, const std::vector<float> &pcm,
                                          size_t crop_offset = 0) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override;
    int mel_bins() const override { return 0; }
    int sample_rate() const override { return 24000; }
    bool is_waveform_encoder() const override { return false; }
    bool has_waveform_embeddings() const override { return true; }

  private:
    void load_arch_hparams() override;
    void load_arch_tensors() override;
    ggml_tensor *encode_text(Graph &g, ggml_tensor *x) const;
    ggml_tensor *adapter(Graph &g, ggml_tensor *x) const;
    ggml_tensor *biasless_norm(Graph &g, ggml_tensor *x, ggml_tensor *weight) const;
    ggml_tensor *project(Graph &g, ggml_tensor *x, const LayerWeights &weights) const;
    const ModelLoader &loader;
    MuqModel audio_backbone;
    nlohmann::json config, text_config;
    struct Parameters {
        int projection, positions, vocab, heads, adapter_heads, head_dim;
        size_t clip_seconds;
        float epsilon;
    } h;
    EmbeddingWeights embeddings;
    std::vector<TransformerWeights> text_layers;
    struct AdapterLayer {
        ggml_tensor *norm, *ffn_norm, *q_scale, *k_scale;
        LayerWeights query, key_value, output, ffn_input, ffn_output;
    };
    std::vector<AdapterLayer> adapter_layers;
    LayerWeights text_projection, audio_projection, audio_latent;
};
} // namespace clap::detail
