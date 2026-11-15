/* Copyright 2026 clap.cpp Authors. Licensed under Apache-2.0. */
#pragma once

#include "model.h"

#include <nlohmann/json.hpp>

namespace clap::detail {

// VoiceNet/voiceclap-large: Qwen2.5-Omni audio encoder and shared Thinker
// decoder. Text IDs are already chat-templated by the vocabulary wrapper.
class VoiceClapModel final : public ModelArchitecture {
  public:
    explicit VoiceClapModel(const ModelLoader &);
    static TensorLoadPolicy loading_policy(const ModelLoader &);
    static bool tensor_quantizable(const ggml_tensor *);
    void initialize_context(Context &) const override;
    std::vector<float> text(Context &, const std::vector<int> &) const override;
    AudioFrames audio_frames(Context &, const std::vector<float> &) const override;
    std::vector<float> waveform_embedding(Context &, const std::vector<float> &, size_t = 0) const override;
    TextParameters text_parameters() const override;
    int projection_dimension() const override { return h.hidden; }
    int mel_bins() const override { return h.mels; }
    int sample_rate() const override { return 16000; }
    bool is_waveform_encoder() const override { return false; }
    bool has_text_encoder() const override { return true; }
    bool has_waveform_embeddings() const override { return true; }

  private:
    friend struct FeatureTestAccess;
    const ModelLoader &loader;
    nlohmann::json config, preprocessor;
    struct Parameters {
        int hidden, heads, kv_heads, head_dim, intermediate, vocab, context, text_layers;
        int audio_hidden, audio_heads, audio_intermediate, audio_layers, mels, window, audio_positions;
        int bos, eos, audio_bos, audio_eos, audio_token, max_samples;
        float epsilon, theta;
    } h{};
    struct DecoderLayer {
        AttentionWeights attention;
        LayerWeights attention_norm, ffn_norm, gate, up, down;
    };
    std::vector<DecoderLayer> decoder_layers;
    std::vector<TransformerWeights> encoder_layers;
    ggml_tensor *token_embedding = nullptr;
    LayerWeights final_norm, conv1, conv2, audio_norm, audio_projection;
    std::vector<int> audio_prefix, audio_suffix;
    void load_arch_hparams() override;
    void load_arch_tensors() override;
    void validate_ids(const std::vector<int> &) const;
    ggml_tensor *decode(Graph &, ggml_tensor *) const;
    ggml_tensor *audio_chunk(Graph &, ggml_tensor *) const;
    HostMatrix audio_tokens(Context &, const HostMatrix &) const;
    static HostMatrix frontend(const std::vector<float> &, int mels = 128, int max_samples = 4800000);
    std::vector<float> audio_features_embedding(Context &, const HostMatrix &) const;
};
} // namespace clap::detail
