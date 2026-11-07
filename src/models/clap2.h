#pragma once
#include "model.h"
#include "graph.h"
#include <nlohmann/json.hpp>

namespace clap::detail {
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
} // namespace clap::detail
