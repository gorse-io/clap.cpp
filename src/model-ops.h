#pragma once

#include "model-loader.h"
#include "impl.h"
#include <cstddef>
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

// Reusable CPU ggml execution primitives. Architectural attention and projection
// composition belongs to each implementation under src/models, not this layer.
class TensorOps {
public:
    TensorOps(const ModelLoader &loader, int threads);
    Matrix linear(const Matrix &x, const std::string &name, bool bias = true) const;
    Matrix conv1d(const Matrix &x, const std::string &name, int kernel, int stride,
                  int padding, int groups, bool bias = true) const;
    Matrix norm(const Matrix &x, const std::string &name, float epsilon) const;
    int threads() const { return n_threads; }
private:
    const ModelLoader &loader;
    int n_threads;
};
} // namespace clap::detail
