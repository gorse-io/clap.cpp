#pragma once
#include "model-ops.h"

namespace clap::detail {
// Native Gemma4-audio Conformer used by EmbeddingGemma2 mmproj GGUFs.
// The non-owning loader must outlive this encoder. No file decoding/resampling.
class GemmaAudioEncoder {
public:
    GemmaAudioEncoder(const ModelLoader &loader, int threads);
    Matrix encode(const std::vector<float> &pcm16k) const;
    static constexpr int sample_rate() { return 16000; }
private:
    friend struct GemmaAudioTestAccess;
    static Matrix log_mel(const std::vector<float> &pcm16k);
    Matrix encode_features(const Matrix &features) const;
    Matrix rms(const Matrix &x, const std::string &name) const;
    Matrix clipped_linear(const Matrix &x, const std::string &name) const;
    Matrix feed_forward(const Matrix &x, const std::string &prefix, const std::string &suffix) const;
    Matrix attention(const Matrix &x, const std::string &prefix, const Matrix &positions) const;
    Matrix subsample(const Matrix &features) const;
    const ModelLoader &loader;
    int hidden, heads, layers, feedforward, channels[2], output;
    float epsilon;
    TensorOps ops;
};
} // namespace clap::detail
