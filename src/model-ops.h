#pragma once

#include "model-loader.h"
#include "impl.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace clap::detail {

struct Matrix {
    int rows, cols;
    std::vector<float> data;
    Matrix(int r, int c);
    float *operator[](int r) { return data.data() + static_cast<size_t>(r) * cols; }
    const float *operator[](int r) const { return data.data() + static_cast<size_t>(r) * cols; }
};
void add(Matrix &a, const Matrix &b);
void gelu(Matrix &x);
float scalar(const ggml_tensor *tensor, size_t index);
// Decode a contiguous embedding row once, not once per channel.
void tensor_row(const ggml_tensor *tensor, int64_t row, float *output);

// Reusable ggml execution primitives (optional Vulkan linear, CPU fallback).
// Architectural attention and projection
// composition belongs to each implementation under src/models, not this layer.
class TensorOps {
public:
    TensorOps(const ModelLoader &loader, int threads);
    Matrix linear(const Matrix &x, const std::string &name, bool bias = true) const;
    Matrix norm(const Matrix &x, const std::string &name, float epsilon) const;
    int threads() const { return n_threads; }
    // Diagnostics report successful execution, not merely device availability.
    bool last_linear_used_vulkan() const;
    size_t vulkan_weight_uploads() const;
private:
    const ModelLoader &loader;
    int n_threads;
    struct BackendState;
    std::shared_ptr<BackendState> backend;
};
} // namespace clap::detail
