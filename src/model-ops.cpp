#include "model-ops.h"
#include <ggml-cpu.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>

namespace clap::detail {

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0) {
    check(r > 0 && c > 0, "invalid matrix shape");
}
void add(Matrix &a, const Matrix &b) {
    check(a.rows == b.rows && a.cols == b.cols, "residual shape mismatch");
    for (size_t i = 0; i < a.data.size(); ++i) a.data[i] += b.data[i];
}
void gelu(Matrix &x) {
    for (float &v : x.data) v = 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f)));
}
float scalar(const ggml_tensor *t, size_t i) {
    if (t->type == GGML_TYPE_F32) return static_cast<const float *>(t->data)[i];
    return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(t->data)[i]);
}
TensorOps::TensorOps(const ModelLoader &source, int threads) : loader(source),
    n_threads(threads > 0 ? threads : static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())))) {}

Matrix TensorOps::linear(const Matrix &x, const std::string &name, bool bias) const {
    auto *w = loader.weight(name + ".weight");
    check(w->ne[0] == x.cols && w->ne[2] == 1 && w->ne[3] == 1, "linear shape: " + name);
    Matrix result(x.rows, static_cast<int>(w->ne[1]));
    const size_t bytes = (x.data.size() + result.data.size()) * sizeof(float) + 2 * 1024 * 1024;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
    check(ctx != nullptr, "ggml allocation failed");
    auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, x.cols, x.rows);
    std::memcpy(input->data, x.data.data(), x.data.size() * sizeof(float));
    auto *output = ggml_mul_mat(ctx.get(), w, input);
    auto *graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    check(ggml_graph_compute_with_ctx(ctx.get(), graph, n_threads) == GGML_STATUS_SUCCESS, "ggml matrix multiply failed");
    std::memcpy(result.data.data(), output->data, result.data.size() * sizeof(float));
    if (bias) {
        const auto *b = loader.weight(name + ".bias");
        for (int r = 0; r < result.rows; ++r) for (int c = 0; c < result.cols; ++c) result[r][c] += scalar(b, c);
    }
    return result;
}

Matrix TensorOps::norm(const Matrix &x, const std::string &name, float eps) const {
    const auto *w = loader.weight(name + ".weight"), *b = loader.weight(name + ".bias");
    Matrix y(x.rows, x.cols);
    for (int r = 0; r < x.rows; ++r) {
        double mean = 0, var = 0;
        for (int c = 0; c < x.cols; ++c) mean += x[r][c];
        mean /= x.cols;
        for (int c = 0; c < x.cols; ++c) { const double d = x[r][c] - mean; var += d * d; }
        const float scale = static_cast<float>(1 / std::sqrt(var / x.cols + eps));
        for (int c = 0; c < x.cols; ++c) y[r][c] = (x[r][c] - static_cast<float>(mean)) * scale * scalar(w, c) + scalar(b, c);
    }
    return y;
}
} // namespace clap::detail
