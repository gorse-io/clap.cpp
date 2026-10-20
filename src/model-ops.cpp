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
    if (t->type == GGML_TYPE_F16) return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(t->data)[i]);
    check(t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q6_K,
          "unsupported scalar tensor type");
    // Decode only the containing block (embeddings, relative biases, or norms),
    // never expand an entire mapped weight tensor/model.
    float block[256];
    const size_t width = static_cast<size_t>(ggml_blck_size(t->type));
    const auto *data = static_cast<const uint8_t *>(t->data) + (i / width) * ggml_type_size(t->type);
    ggml_get_type_traits(t->type)->to_float(data, block, width);
    return block[i % width];
}
void tensor_row(const ggml_tensor *t, int64_t row, float *output) {
    check(row >= 0 && row < ggml_nrows(t), "embedding row out of range");
    const auto *data = static_cast<const uint8_t *>(t->data) + row * t->nb[1];
    if (t->type == GGML_TYPE_F32) std::copy_n(reinterpret_cast<const float *>(data), t->ne[0], output);
    else if (t->type == GGML_TYPE_F16)
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(data), output, t->ne[0]);
    else {
        const auto *traits = ggml_get_type_traits(t->type);
        check(traits && traits->to_float, "unsupported embedding tensor type");
        traits->to_float(data, output, t->ne[0]);
    }
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
    auto plan = ggml_graph_plan(graph, n_threads, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    check(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "ggml matrix multiply failed");
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
