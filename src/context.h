#pragma once

#include "model.h"

#include <memory>
#include <unordered_map>

namespace clap::detail {

// Mutable execution state. Model ownership precedes runtimes so cached graphs,
// schedulers and compute backends are destroyed before shared model resources.
class Context {
  public:
    Context(std::shared_ptr<const Model>, int threads = 0);
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    const Model &model() const;
    GraphRuntime &runtime(const ModelArchitecture *, const ModelLoader &, bool preserve = false);

  private:
    friend struct FeatureTestAccess;
    // The raw-feature oracle bridge supplies the loader/architecture lifetime.
    Context(std::shared_ptr<ModelWeights>, int threads);
    std::shared_ptr<const Model> owner;
    std::shared_ptr<ModelWeights> weights;
    int threads;
    struct Slot {
        const ModelLoader *loader;
        bool preserve;
        std::unique_ptr<GraphRuntime> execution;
    };
    std::unordered_map<const ModelArchitecture *, Slot> runtimes;
};
} // namespace clap::detail

namespace clap::detail {
// Core-only operations shared by the C entry points. Upper layers use clap.h.
std::vector<int> tokenize(const Model &, const std::string &utf8, size_t max_length);
std::vector<float> encode_text(Context &, const std::string &utf8);
std::vector<float> encode_audio(Context &, const std::vector<float> &pcm, int sample_rate, size_t crop_offset);
std::vector<float> encode_audio_file(Context &, const std::string &path, size_t crop_offset);
AudioFrames audio_frames(Context &, const std::vector<float> &pcm, int sample_rate);
} // namespace clap::detail
