#include "model-ops.h"
#include <stdexcept>
#include <ggml-cpu.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
#ifdef CLAP_USE_VULKAN
#include <ggml-backend.h>
#include <ggml-vulkan.h>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#endif

namespace clap::detail {
namespace {
// Storage-only precision: retain mapped bytes, decode just the current dense
// operand, and never insert this temporary alias in the persistent GPU cache.
struct MatmulWeight {
    std::vector<float> values;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    ggml_tensor *tensor;
    MatmulWeight(ggml_tensor *weight, bool preserve_inputs) : tensor(weight) {
        if (!preserve_inputs || weight->type == GGML_TYPE_F32) return;
        const auto *traits = ggml_get_type_traits(weight->type);
        if (!ggml_is_contiguous(weight) || !traits || !traits->to_float)
            throw std::runtime_error("CLAP: unsupported storage-only matmul operand");
        values.resize(static_cast<size_t>(ggml_nelements(weight)));
        traits->to_float(weight->data, values.data(), ggml_nelements(weight));
        context.reset(ggml_init({4096, nullptr, true}));
        if (!context) throw std::runtime_error("CLAP: matmul alias allocation failed");
        tensor = ggml_new_tensor(context.get(), GGML_TYPE_F32, ggml_n_dims(weight), weight->ne);
        tensor->data = values.data();
    }
};
} // namespace

struct TensorOps::BackendState {
#ifdef CLAP_USE_VULKAN
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> gpu{nullptr, ggml_backend_free};
    struct Weight {
        Context ctx{nullptr, ggml_free};
        Buffer buffer{nullptr, ggml_backend_buffer_free};
        ggml_tensor *tensor = nullptr;
    };
    // Declared after gpu: device buffers must be destroyed before the backend.
    std::unordered_map<const ggml_tensor *, std::unique_ptr<Weight>> weights;
    mutable std::mutex mutex;
    bool last_vulkan = false;
    bool enabled = true;
    size_t uploads = 0;

    BackendState() {
        const auto *setting = std::getenv("CLAP_VULKAN");
        if (setting && std::strcmp(setting, "0") == 0) return;
        try {
            if (ggml_backend_vk_get_device_count() > 0) gpu.reset(ggml_backend_vk_init(0));
        } catch (const std::exception &) {
            // An optional backend must not prevent loading on a CPU-only host.
        }
    }

    bool linear(const Matrix &x, const ggml_tensor *w, Matrix &result) {
        if (!gpu || !enabled) return false;
        Context ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
        if (!ctx) return false;
        // The pinned Vulkan cooperative-matrix path loses partial output tiles
        // on llvmpipe (e.g. M=37, N=9: columns 32..36 of the last row).
        // Pad both non-reduction dimensions to native cooperative tile boundaries.
        // Zero channels/rows do not change real results. Keep <=8-row matvec
        // batches unpadded: those kernels do not use the broken tiled path.
        const int64_t input_rows = x.rows > 8 ? (int64_t(x.rows) + 31) / 32 * 32 : x.rows;
        auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, x.cols, input_rows);
        auto found = weights.find(w);
        std::unique_ptr<Weight> pending;
        Weight *cached;
        if (found == weights.end()) {
            pending = std::make_unique<Weight>();
            pending->ctx.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
            if (!pending->ctx) return false;
            pending->tensor = ggml_new_tensor_2d(pending->ctx.get(), w->type, w->ne[0], (w->ne[1] + 31) / 32 * 32);
            cached = pending.get();
        } else {
            cached = found->second.get();
        }
        auto *output = ggml_mul_mat(ctx.get(), cached->tensor, input);
        // Match the CPU's float accumulation even when weights are half storage.
        // GGML_PREC_DEFAULT permits half accumulation in Vulkan's F16 kernels.
        ggml_prec_set_acc(output, GGML_PREC_F32);
        const auto device = ggml_backend_get_device(gpu.get());
        if (!ggml_backend_dev_supports_op(device, output)) return false;
        if (pending) {
            cached->buffer.reset(ggml_backend_alloc_ctx_tensors(cached->ctx.get(), gpu.get()));
            if (!cached->buffer) return false;
            ggml_backend_buffer_set_usage(cached->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            const size_t weight_bytes = ggml_nbytes(w);
            const size_t padding_bytes = ggml_nbytes(cached->tensor) - weight_bytes;
            if (padding_bytes) ggml_backend_tensor_memset(cached->tensor, 0, weight_bytes, padding_bytes);
            ggml_backend_tensor_set(cached->tensor, w->data, 0, weight_bytes);
            weights.emplace(w, std::move(pending));
            ++uploads;
        }
        auto *graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), gpu.get()), ggml_backend_buffer_free);
        if (!buffer) return false;
        const size_t input_bytes = x.data.size() * sizeof(float);
        if (input_rows != x.rows) ggml_backend_tensor_memset(input, 0, input_bytes, ggml_nbytes(input) - input_bytes);
        ggml_backend_tensor_set(input, x.data.data(), 0, input_bytes);
        if (ggml_backend_graph_compute(gpu.get(), graph) != GGML_STATUS_SUCCESS) return false;
        // The backend may optimize mul_mat to a transposed/strided result.
        // tensor_get uses device transfers; never dereference output->data.
        if (ggml_is_contiguous(output) && output->ne[0] == result.cols) {
            ggml_backend_tensor_get(output, result.data.data(), 0, result.data.size() * sizeof(float));
        } else {
            std::vector<unsigned char> storage(ggml_nbytes(output));
            ggml_backend_tensor_get(output, storage.data(), 0, storage.size());
            for (int r = 0; r < result.rows; ++r) for (int c = 0; c < result.cols; ++c)
                std::memcpy(&result[r][c], storage.data() + size_t(r) * output->nb[1] +
                            size_t(c) * output->nb[0], sizeof(float));
        }
        return true;
    }
#endif
};

bool TensorOps::last_linear_used_vulkan() const {
#ifdef CLAP_USE_VULKAN
    std::lock_guard<std::mutex> lock(backend->mutex);
    return backend->last_vulkan;
#else
    return false;
#endif
}
size_t TensorOps::vulkan_weight_uploads() const {
#ifdef CLAP_USE_VULKAN
    std::lock_guard<std::mutex> lock(backend->mutex);
    return backend->uploads;
#else
    return 0;
#endif
}

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0) {
    if (!(r > 0 && c > 0))
        throw std::runtime_error("CLAP: invalid matrix shape");
}
void add(Matrix &a, const Matrix &b) {
    if (!(a.rows == b.rows && a.cols == b.cols))
        throw std::runtime_error("CLAP: residual shape mismatch");
    for (size_t i = 0; i < a.data.size(); ++i) a.data[i] += b.data[i];
}
void gelu(Matrix &x) {
    for (float &v : x.data) v = 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f)));
}
float scalar(const ggml_tensor *t, size_t i) {
    if (t->type == GGML_TYPE_F32) return static_cast<const float *>(t->data)[i];
    if (t->type == GGML_TYPE_F16) return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(t->data)[i]);
    if (t->type == GGML_TYPE_BF16) return ggml_bf16_to_fp32(static_cast<const ggml_bf16_t *>(t->data)[i]);
    if (!(t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K || t->type == GGML_TYPE_Q6_K))
        throw std::runtime_error("CLAP: unsupported scalar tensor type");
    // Decode only the containing block (embeddings, relative biases, or norms),
    // never expand an entire mapped weight tensor/model.
    float block[256];
    const size_t width = static_cast<size_t>(ggml_blck_size(t->type));
    const auto *data = static_cast<const uint8_t *>(t->data) + (i / width) * ggml_type_size(t->type);
    ggml_get_type_traits(t->type)->to_float(data, block, width);
    return block[i % width];
}
void tensor_row(const ggml_tensor *t, int64_t row, float *output) {
    if (!(row >= 0 && row < ggml_nrows(t)))
        throw std::runtime_error("CLAP: embedding row out of range");
    const auto *data = static_cast<const uint8_t *>(t->data) + row * t->nb[1];
    if (t->type == GGML_TYPE_F32) std::copy_n(reinterpret_cast<const float *>(data), t->ne[0], output);
    else if (t->type == GGML_TYPE_F16)
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(data), output, t->ne[0]);
    else {
        const auto *traits = ggml_get_type_traits(t->type);
        if (!(traits && traits->to_float))
            throw std::runtime_error("CLAP: unsupported embedding tensor type");
        traits->to_float(data, output, t->ne[0]);
    }
}
TensorOps::TensorOps(const ModelLoader &source, int threads) : loader(source),
    n_threads(threads > 0 ? threads : static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())))) {
#ifdef CLAP_USE_VULKAN
    backend = std::make_shared<BackendState>();
#endif
}

Matrix TensorOps::linear(const Matrix &x, const std::string &name, bool bias) const {
    auto *w = loader.weight(name + ".weight");
    if (!(w->ne[0] == x.cols && w->ne[2] == 1 && w->ne[3] == 1))
        throw std::runtime_error("CLAP: linear shape: " + name);
    MatmulWeight operand(w, loader.f32_activation_compute());
    Matrix result(x.rows, static_cast<int>(w->ne[1]));
    bool computed = false;
#ifdef CLAP_USE_VULKAN
    std::lock_guard<std::mutex> lock(backend->mutex);
    backend->last_vulkan = false;
    try {
        if (!loader.f32_activation_compute()) computed = backend->linear(x, w, result);
        backend->last_vulkan = computed;
    } catch (const std::exception &) {
        // Allocation/driver failures are recoverable: use the original CPU path.
        backend->enabled = false;
    }
#endif
    if (!computed) {
        const size_t bytes = (x.data.size() + result.data.size()) * sizeof(float) + 2 * 1024 * 1024;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
        if (ctx == nullptr)
            throw std::runtime_error("CLAP: ggml allocation failed");
        auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, x.cols, x.rows);
        std::memcpy(input->data, x.data.data(), x.data.size() * sizeof(float));
        auto *output = ggml_mul_mat(ctx.get(), operand.tensor, input);
        auto *graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        auto plan = ggml_graph_plan(graph, n_threads, nullptr);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.data();
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CLAP: ggml matrix multiply failed");
        std::memcpy(result.data.data(), output->data, result.data.size() * sizeof(float));
    }
    if (bias) {
        const auto *b = loader.weight(name + ".bias");
        for (int r = 0; r < result.rows; ++r) for (int c = 0; c < result.cols; ++c) result[r][c] += scalar(b, c);
    }
    return result;
}

Matrix TensorOps::conv1d(const Matrix &x, const std::string &name, int kernel,
                         int stride, int padding, int groups, bool bias) const {
    const auto *w = loader.weight(name + ".weight");
    if (!(kernel > 0 && stride > 0 && padding >= 0 && groups > 0))
        throw std::runtime_error("CLAP: invalid convolution parameters: " + name);
    if (!(w->ne[0] == kernel && w->ne[1] * groups == x.cols && w->ne[2] > 0 &&
          w->ne[2] % groups == 0 && w->ne[3] == 1)) throw std::runtime_error("CLAP: convolution shape: " + name);
    const int rows = (x.rows + 2 * padding - kernel) / stride + 1;
    if (!(rows > 0))
        throw std::runtime_error("CLAP: convolution input is too short: " + name);
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
        if (alias_context == nullptr)
            throw std::runtime_error("CLAP: ggml convolution alias allocation failed");
        auto *flat = ggml_new_tensor_2d(alias_context.get(), w->type, x.cols, output_columns);
        flat->data = w->data;
        MatmulWeight operand(flat, loader.f32_activation_compute());
        const size_t bytes = (x.data.size() + static_cast<size_t>(rows) * output_columns) * sizeof(float) +
                             2 * 1024 * 1024;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
        if (ctx == nullptr)
            throw std::runtime_error("CLAP: ggml convolution allocation failed");
        auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, x.cols, x.rows);
        std::memcpy(input->data, x.data.data(), x.data.size() * sizeof(float));
        auto *output = ggml_mul_mat(ctx.get(), operand.tensor, input);
        auto *graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        if (ggml_graph_compute_with_ctx(ctx.get(), graph, n_threads) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CLAP: ggml convolution matrix multiply failed");
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
