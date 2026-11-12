#pragma once
#include "graph.h"

namespace clap::detail {
// Native Gemma4-audio Conformer used by EmbeddingGemma2 mmproj GGUFs.
// The non-owning loader must outlive this encoder. No file decoding/resampling.
class GemmaAudioEncoder {
  public:
    GemmaAudioEncoder(const ModelLoader &loader, int threads);
    HostMatrix encode(const std::vector<float> &pcm16k) const;
    static constexpr int sample_rate() { return 16000; }

  private:
    friend struct GemmaAudioTestAccess;
    static HostMatrix log_mel(const std::vector<float> &pcm16k);
    HostMatrix encode_features(const HostMatrix &features) const;
    HostMatrix rms(const HostMatrix &x, const std::string &name) const;
    HostMatrix clipped_linear(const HostMatrix &x, const std::string &name) const;
    HostMatrix feed_forward(const HostMatrix &x, const std::string &prefix, const std::string &suffix) const;
    HostMatrix attention(const HostMatrix &x, const std::string &prefix, const HostMatrix &positions) const;
    HostMatrix subsample(const HostMatrix &features) const;
    const ModelLoader &loader;
    int hidden, heads, layers, feedforward, channels[2], output;
    float epsilon;
    GraphRuntime runtime;
    int threads;
    HostMatrix linear(const HostMatrix &x, const std::string &name, bool bias = false) const;
};
} // namespace clap::detail
