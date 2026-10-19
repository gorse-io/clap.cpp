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

Matrix TensorOps::conv1d(const Matrix &x, const std::string &name, int kernel,
                         int stride, int padding, int groups, bool bias) const {
    const auto *w = loader.weight(name + ".weight");
    check(kernel > 0 && stride > 0 && padding >= 0 && groups > 0,
          "invalid convolution parameters: " + name);
    check(w->ne[0] == kernel && w->ne[1] * groups == x.cols && w->ne[2] > 0 &&
          w->ne[2] % groups == 0 && w->ne[3] == 1, "convolution shape: " + name);
    const int rows = (x.rows + 2 * padding - kernel) / stride + 1;
    check(rows > 0, "convolution input is too short: " + name);
    const int output_columns = static_cast<int>(w->ne[2]);

    // Conv1d with a one-sample kernel is a time-distributed dense layer. Use
    // ggml's threaded matrix multiply for the large MERT pointwise blocks.
    if (kernel == 1 && stride == 1 && padding == 0 && groups == 1) {
        // The stored Conv1d [out,in,1] bytes are identical to a dense
        // [out,in] matrix. Keep the alias in a no-allocation metadata context;
        // allocating its data in the scratch graph would duplicate multi-MiB
        // published weights and can exhaust the deliberately small arena.
        std::unique_ptr<ggml_context, decltype(&ggml_free)> alias_context(
            ggml_init({4096, nullptr, true}), ggml_free);
        check(alias_context != nullptr, "ggml convolution alias allocation failed");
        auto *flat = ggml_new_tensor_2d(alias_context.get(), w->type, x.cols, output_columns);
        flat->data = w->data;
        const size_t bytes = (x.data.size() + static_cast<size_t>(rows) * output_columns) * sizeof(float) +
                             2 * 1024 * 1024;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
        check(ctx != nullptr, "ggml convolution allocation failed");
        auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, x.cols, x.rows);
        std::memcpy(input->data, x.data.data(), x.data.size() * sizeof(float));
        auto *output = ggml_mul_mat(ctx.get(), flat, input);
        auto *graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        check(ggml_graph_compute_with_ctx(ctx.get(), graph, n_threads) == GGML_STATUS_SUCCESS,
              "ggml convolution matrix multiply failed");
        Matrix result(rows, output_columns);
        std::memcpy(result.data.data(), output->data, result.data.size() * sizeof(float));
        if (bias) {
            const auto *b = loader.weight(name + ".bias");
            for (int row = 0; row < rows; ++row)
                for (int column = 0; column < output_columns; ++column)
                    result[row][column] += scalar(b, column);
        }
        return result;
    }

    Matrix result(rows, output_columns);
    const int inputs_per_group = x.cols / groups;
    const int outputs_per_group = output_columns / groups;
    const ggml_tensor *b = bias ? loader.weight(name + ".bias") : nullptr;
    for (int row = 0; row < rows; ++row) for (int output = 0; output < output_columns; ++output) {
        float sum = b ? scalar(b, output) : 0.0f;
        const int group = output / outputs_per_group;
        for (int input = 0; input < inputs_per_group; ++input) for (int tap = 0; tap < kernel; ++tap) {
            const int source = row * stride + tap - padding;
            if (source >= 0 && source < x.rows) {
                const size_t index = (static_cast<size_t>(output) * inputs_per_group + input) * kernel + tap;
                sum += x[source][group * inputs_per_group + input] * scalar(w, index);
            }
        }
        result[row][output] = sum;
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
