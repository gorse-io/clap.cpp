#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct BenchOptions {
    std::string model_path;
    std::string text;
    std::string audio_path;
    std::string device;
    int threads = 0;
    std::size_t crop_offset = 0;
    bool crop_requested = false;
    bool frames = false;
    std::vector<std::string> benchmark_args;
};

int cmd_bench(const BenchOptions &options);
