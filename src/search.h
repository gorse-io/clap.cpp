#pragma once

#include <cstddef>
#include <string>

struct SearchOptions {
    std::string directory;
    std::string model_path;
    std::string text;
    std::string audio_path;
    std::size_t top_k = 10;
    int threads = 0;
    std::string device;
    bool text_requested = false;
};

int cmd_search(const SearchOptions &options);
