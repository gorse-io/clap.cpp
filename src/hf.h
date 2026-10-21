#pragma once

#include <string>
#include <vector>

namespace clap {
struct HfOptions {
    std::string repo;
    std::string file;
    std::string token;
    bool offline = false;
};

// Resolve a single, unsplit GGUF into the Hugging Face-compatible cache.
// Throws std::runtime_error on invalid input, network or cache errors.
std::string hf_download(const HfOptions &options);

// CLI11 does not accept multi-character short options. Normalize only the
// documented llama.cpp-style aliases before parsing (including =value forms).
std::vector<std::string> hf_arguments(int argc, char **argv);
} // namespace clap
