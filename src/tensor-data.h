#pragma once
#include <ggml.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace clap::detail {
// Host storage for audio frontends and public API boundaries.
struct HostMatrix {
    int rows, cols;
    std::vector<float> data;
    HostMatrix(int rows, int cols);
    float *operator[](int row) { return data.data() + size_t(row) * cols; }
    const float *operator[](int row) const { return data.data() + size_t(row) * cols; }
};
float scalar(const ggml_tensor *tensor, size_t index);
void tensor_row(const ggml_tensor *tensor, int64_t row, float *output);
}
