#pragma once
// Test-only access to private tensor-input encoders and MERT feature extraction;
// never installed or linked into the production CLI. Keeps numerical oracles
// independent of the bundled waveform pipeline.
#include "../../src/context.h"
#include "../../src/models/models.h"
#include "audio.h"

#include <stdexcept>
namespace clap::detail {
struct FeatureTestAccess {
    // Deliberately construct MERT directly: this is not ModelArchitecture dispatch.
    // Match the bundled frontend's 24 kHz input contract for independent oracles.
    static AudioFrames extract(const std::string &path, const std::vector<float> &pcm, int sample_rate = 24000,
                               const RuntimeOptions &options = {}) {
        if (sample_rate <= 0)
            throw std::runtime_error("test feature extraction: audio sample rate must be positive");
        ModelLoader loader(options);
        clap::detail::load_model_weights(loader, path);
        if (loader.metadata("general.architecture") != "mert")
            throw std::runtime_error("test feature extraction requires a standalone MERT model");
        auto weights = std::make_shared<ModelWeights>(loader);
        MertModel model(loader);
        Context context(weights, 2);
        model.initialize_context(context);
        if (sample_rate == 24000)
            return model.clamp3_features(context, pcm);
        return model.clamp3_features(context, resample_audio({pcm, sample_rate}, 24000).samples);
    }
    static std::vector<float> encode(const std::string &path, const std::vector<float> &values, int frames,
                                     const RuntimeOptions &options = {}) {
        ModelLoader loader(options);
        clap::detail::load_model_weights(loader, path);
        auto weights = std::make_shared<ModelWeights>(loader);
        if (loader.metadata("general.architecture") == "clap") {
            ClapModel model(loader);
            Context context(weights, 2);
            model.initialize_context(context);
            return model.encode_features(context, values, frames);
        }
        Clamp3Model model(loader);
        Context context(weights, 2);
        model.initialize_context(context);
        return model.encode_features(context, values, frames);
    }
};
} // namespace clap::detail
