#include "feature-test.h"

#include <iostream>

// Test-only numerical-oracle bridge. Not part of the production CLI or install.
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::runtime_error("usage: test-feature-extract MERT.gguf AUDIO");
        const auto audio = clap::decode_audio(argv[2]);
        const auto features = clap::detail::FeatureTestAccess::extract(argv[1], audio.samples, audio.sample_rate);
        std::cout << nlohmann::json(
                         {{"values", features.values}, {"frames", features.frames}, {"dimension", features.dimension}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
