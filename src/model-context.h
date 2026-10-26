#pragma once

#include "model.h"
#include "model-ops.h"

namespace clap::detail {

// Shared execution state for architecture-specific encoders, corresponding to
// llama.cpp's common graph context. This runtime executes TensorOps eagerly:
// it is not a persistent GGML graph. Keep the context alive across calls so
// backend selection and uploaded-weight caches retain their existing lifetime.
// The model's validated parameters and loader must outlive this context.
class ModelContext {
public:
    ModelContext(const ModelLoader &source, int threads) : loader(source), ops(source, threads) {}
    virtual ~ModelContext() = default;
    ModelContext(const ModelContext &) = delete;
    ModelContext &operator=(const ModelContext &) = delete;

    virtual std::vector<float> text(const std::vector<int> &ids) const = 0;
    virtual AudioFrames audio_frames(const std::vector<float> &pcm) const = 0;
    virtual AudioFrames clamp3_features(const std::vector<float> &) const {
        throw std::runtime_error("CLaMP3 feature extraction requires a MERT-v1 model");
    }
    virtual std::vector<float> waveform_embedding(const std::vector<float> &, size_t = 0) const {
        throw std::runtime_error("CLAP: this architecture has no waveform embeddings");
    }

protected:
    const ModelLoader &loader;
    TensorOps ops;
};
} // namespace clap::detail
