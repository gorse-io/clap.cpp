#pragma once
// Test-only access to private tensor-input encoders; never installed or linked
// into the production CLI. Keeps numerical oracles independent of waveform input.
#include "../../src/models/models.h"
namespace clap::detail {
struct FeatureTestAccess {
    static std::vector<float> encode(const std::string &path, const std::vector<float> &values, int frames) {
        ModelLoader loader; loader.load(path);
        if (loader.metadata("general.architecture") == "clap") {
            ClapModel model(loader, 2); return model.encode_features(values, frames);
        }
        Clamp3Model model(loader, 2); return model.encode_features(values, frames);
    }
};
}
