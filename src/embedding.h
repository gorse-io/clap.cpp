#pragma once

#include <cstddef>
#include <string>

struct EmbeddingOptions {
    std::string model_path;
    int threads = 0;
    std::string text;
    std::string audio_path;
    std::string separator = "\n";
    // Empty means native single-vector JSON for --text/--audio, raw for --prompt/--file.
    std::string format;
    bool single_text = false;
    std::size_t crop_offset = 0;
    bool crop_requested = false;
    bool frames = false;
    bool clamp3_features = false;
};

int run_embedding(const EmbeddingOptions &options);
