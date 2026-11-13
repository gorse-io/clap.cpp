#include "graph.h"
#include "model.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
namespace clap::detail {
HostMatrix::HostMatrix(int r, int c) : rows(r), cols(c), data() {
    if (!(r > 0 && c > 0))
        throw std::runtime_error("CLAP: invalid matrix shape");
    data.resize(static_cast<size_t>(r) * c);
}

namespace {
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
} // namespace
struct GraphRuntime::State {
    const ModelLoader &loader;
    bool f32_activations;
    std::mutex mutex;
    std::shared_ptr<ModelWeights> weights;
    Backend gpu{nullptr, ggml_backend_free};
    Backend cpu{nullptr, ggml_backend_free};
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler{nullptr, ggml_backend_sched_free};
    size_t scheduler_capacity = 0;
    bool vulkan = false, metal = false;
    bool last_gpu = false;
    size_t uploads = 0;
    int threads;
    GraphStats stats;
    std::string cache_key;
    // Destroy cached graph before its scheduler, compute backends and model store.
    std::unique_ptr<Graph> cache;
    State(const ModelLoader &source, int requested_threads, bool preserve_activations,
          std::shared_ptr<ModelWeights> shared_weights)
        : loader(source), f32_activations(preserve_activations || source.f32_activation_compute()),
          weights(shared_weights ? std::move(shared_weights) : std::make_shared<ModelWeights>(source)),
          threads(requested_threads > 0 ? requested_threads : int(std::max(1u, std::thread::hardware_concurrency()))) {
        f32_activations |= source.runtime_options().activation_precision == ActivationPrecision::F32;
        cpu.reset(ggml_backend_cpu_init());
        if (!cpu)
            throw std::runtime_error("CLAP: CPU backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.get(), threads);
        if (auto device = weights->gpu_device()) {
            gpu.reset(ggml_backend_dev_init(device, nullptr));
            if (!gpu)
                throw std::runtime_error("CLAP: cannot initialize requested GGML device: " +
                                         std::string(ggml_backend_dev_name(device)));
        }
        vulkan = weights->uses_vulkan();
        metal = weights->uses_metal();
    }
};
GraphRuntime::GraphRuntime(const ModelLoader &loader, int threads, bool f32_activations,
                           std::shared_ptr<ModelWeights> weights)
    : state(std::make_unique<State>(loader, threads, f32_activations, std::move(weights))) {}
GraphRuntime::~GraphRuntime() = default;
bool GraphRuntime::last_graph_used_vulkan() const {
    std::lock_guard<std::mutex> guard(state->mutex);
    return state->vulkan && state->last_gpu;
}
size_t GraphRuntime::vulkan_weight_uploads() const {
    std::lock_guard<std::mutex> guard(state->mutex);
    return state->vulkan ? state->uploads : 0;
}
GraphStats GraphRuntime::stats() const {
    std::lock_guard<std::mutex> guard(state->mutex);
    auto stats = state->stats;
    stats.device = ggml_backend_dev_name(ggml_backend_get_device(state->gpu ? state->gpu.get() : state->cpu.get()));
    stats.device_matmul = state->last_gpu;
    stats.weight_uploads = state->uploads;
    stats.threads = state->threads;
    return stats;
}
GraphInput::GraphInput(const std::vector<float> &values, int64_t cols, int64_t count)
    : type(GGML_TYPE_F32), columns(cols), rows(count), data(values.data()), bytes(values.size() * sizeof(float)) {
    if (cols <= 0 || count <= 0 || size_t(cols) > values.size() || size_t(count) != values.size() / size_t(cols) ||
        values.size() % size_t(cols))
        throw std::runtime_error("CLAP: invalid dynamic input shape");
}
GraphInput::GraphInput(const std::vector<int> &values)
    : type(GGML_TYPE_I32), columns(values.size()), rows(1), data(values.data()), bytes(values.size() * sizeof(int)) {
    static_assert(sizeof(int) == sizeof(int32_t));
    if (values.empty())
        throw std::runtime_error("CLAP: empty dynamic indices");
}
std::vector<GraphOutput> GraphRuntime::run_cached(const std::string &key, const std::vector<GraphInput> &inputs,
                                                  const std::function<std::vector<ggml_tensor *>(Graph &)> &build,
                                                  size_t extra_nodes) const {
    std::lock_guard<std::mutex> guard(state->mutex);
    bool reuse = state->cache && state->cache_key == key && state->cache->dynamic_inputs.size() == inputs.size();
    if (reuse)
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto *tensor = state->cache->dynamic_inputs[i];
            reuse &=
                tensor->type == inputs[i].type && tensor->ne[0] == inputs[i].columns && tensor->ne[1] == inputs[i].rows;
        }
    try {
        if (!reuse) {
            state->cache.reset();
            auto graph = std::unique_ptr<Graph>(new Graph(*this, extra_nodes, true));
            for (const auto &input : inputs) {
                auto *tensor = ggml_new_tensor_2d(graph->ctx(), input.type, input.columns, input.rows);
                ggml_set_input(tensor);
                graph->dynamic_inputs.push_back(tensor);
            }
            auto outputs = build(*graph);
            if (outputs.empty())
                throw std::runtime_error("CLAP: graph has no outputs");
            for (auto *output : outputs)
                graph->outputs.push_back(ggml_cont(graph->ctx(), output));
            graph->allocate(graph->outputs);
            state->cache_key = key;
            state->cache = std::move(graph);
            ++state->stats.builds;
        } else
            ++state->stats.hits;
        state->cache->update(inputs);
        state->cache->execute();
        std::vector<GraphOutput> result;
        for (auto *output : state->cache->outputs)
            result.push_back({state->cache->read(output), output->ne[0], output->ne[1]});
        return result;
    } catch (...) {
        state->cache.reset();
        throw;
    }
}
Graph::Graph(const GraphRuntime &owner, size_t extra_nodes) : Graph(owner, extra_nodes, false) {}
Graph::Graph(const GraphRuntime &owner, size_t extra_nodes, bool cached)
    : runtime(*owner.state), lock(runtime.mutex, std::defer_lock),
      capacity(std::max(size_t(2048), runtime.loader.tensor_count() * 64) + extra_nodes) {
    if (!cached) {
        lock.lock();
        runtime.cache.reset();
    }
    if (runtime.scheduler)
        ggml_backend_sched_reset(runtime.scheduler.get());
    context.reset(ggml_init(
        {capacity * ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(capacity, false), nullptr, true}));
    if (!context)
        throw std::runtime_error("CLAP: graph metadata allocation failed");
}
Graph::~Graph() {
    if (runtime.scheduler)
        ggml_backend_sched_reset(runtime.scheduler.get());
}
ggml_tensor *Graph::dynamic_input(size_t index) const { return dynamic_inputs.at(index); }
void Graph::update(const std::vector<GraphInput> &values) {
    for (size_t i = 0; i < values.size(); ++i) {
        auto *tensor = dynamic_inputs.at(i);
        if (tensor->buffer)
            ggml_backend_tensor_set(tensor, values[i].data, 0, values[i].bytes);
    }
}

ggml_tensor *Graph::input(const std::vector<float> &data, int64_t columns, int64_t rows) {
    if (columns <= 0 || rows <= 0 || size_t(columns) * size_t(rows) != data.size())
        throw std::runtime_error("CLAP: invalid graph input shape");
    auto *tensor = ggml_new_tensor_2d(ctx(), GGML_TYPE_F32, columns, rows);
    ggml_set_input(tensor);
    Input input{tensor, std::vector<uint8_t>(data.size() * sizeof(float))};
    std::memcpy(input.bytes.data(), data.data(), input.bytes.size());
    inputs.push_back(std::move(input));
    return tensor;
}
ggml_tensor *Graph::indices(const std::vector<int> &data) {
    static_assert(sizeof(int) == sizeof(int32_t));
    if (data.empty())
        throw std::runtime_error("CLAP: empty graph indices");
    auto *tensor = ggml_new_tensor_1d(ctx(), GGML_TYPE_I32, data.size());
    ggml_set_input(tensor);
    Input input{tensor, std::vector<uint8_t>(data.size() * sizeof(int32_t))};
    std::memcpy(input.bytes.data(), data.data(), input.bytes.size());
    inputs.push_back(std::move(input));
    return tensor;
}
ggml_tensor *Graph::raw_weight(ggml_tensor *source, bool dense) {
    const bool device = dense && runtime.gpu;
    bool uploaded = false;
    auto *tensor = runtime.weights->get(source, device ? runtime.gpu.get() : runtime.cpu.get(), device, uploaded);
    runtime.uploads += uploaded ? 1 : 0;
    return tensor;
}
ggml_tensor *Graph::weight(ggml_tensor *source, bool f32) {
    auto *w = raw_weight(source, false);
    if (!f32 || w->type == GGML_TYPE_F32)
        return w;
    auto found = converted.find(source);
    if (found != converted.end())
        return found->second;
    return converted.emplace(source, ggml_cast(ctx(), w, GGML_TYPE_F32)).first->second;
}
ggml_tensor *Graph::matmul(ggml_tensor *weight, ggml_tensor *input, bool offload) {
    // Let ggml select native MM/MMA or MV kernels, as in llama.cpp. Only
    // explicit/model-required F32 activations need the vector-product fallback:
    // Metal's matrix kernels can round operands to F16 despite F32 accumulation.
    if (runtime.metal && runtime.f32_activations && weight->ne[0] >= 64 && input->ne[1] > 1) {
        const int64_t rows = input->ne[1], batches = input->ne[2], groups = input->ne[3];
        // Retain general GGML broadcasting when the two batch axes differ.
        if ((weight->ne[2] == batches && weight->ne[3] == groups) || (weight->ne[2] == 1 && weight->ne[3] == 1)) {
            auto *w = ggml_is_contiguous(weight) ? weight : ggml_cont(ctx(), weight);
            w = ggml_reshape_4d(ctx(), w, w->ne[0], w->ne[1], 1, w->ne[2] * w->ne[3]);
            ggml_tensor *output = nullptr;
            // Metal's batch indices are int16. Tile large sequences within that bound.
            for (int64_t first = 0; first < rows; first += 16384) {
                const int64_t count = std::min(int64_t(16384), rows - first);
                auto *tile = ggml_view_4d(ctx(), input, input->ne[0], count, batches, groups, input->nb[1],
                                          input->nb[2], input->nb[3], first * input->nb[1]);
                tile = ggml_reshape_4d(ctx(), ggml_cont(ctx(), tile), input->ne[0], 1, count, batches * groups);
                auto *product = ggml_mul_mat(ctx(), w, tile);
                if (offload && runtime.gpu && ggml_backend_supports_op(runtime.gpu.get(), product))
                    device_nodes.push_back(product);
                ggml_prec_set_acc(product, GGML_PREC_F32);
                ggml_prec_set_src(product, GGML_PREC_F32, 1);
                product = ggml_reshape_4d(ctx(), product, weight->ne[1], count, batches, groups);
                output = output ? ggml_concat(ctx(), output, product, 1) : product;
            }
            return output;
        }
    }
    auto *result = ggml_mul_mat(ctx(), weight, input);
    if (offload && runtime.gpu && ggml_backend_supports_op(runtime.gpu.get(), result))
        device_nodes.push_back(result);
    ggml_prec_set_acc(result, GGML_PREC_F32);
    if (runtime.f32_activations || weight->type == GGML_TYPE_F32)
        ggml_prec_set_src(result, GGML_PREC_F32, 1);
    return result;
}
ggml_tensor *Graph::linear(ggml_tensor *x, const LayerWeights &weights) {
    const std::string name = ggml_get_name(weights.weight);
    ggml_tensor *w;
    if ((runtime.f32_activations || weights.weight->type == GGML_TYPE_BF16) && weights.weight->type != GGML_TYPE_F32) {
        // BF16 native matmul also packs F32 activations to BF16. Decode the
        // current operand in the graph instead of expanding the mapped model.
        // Keep casts on CPU: some GPU casts round decoded weights through F16.
        w = weight(weights.weight);
        cpu_nodes.push_back(w);
    } else
        w = raw_weight(weights.weight, true);
    if (x->ne[0] != w->ne[0] || w->ne[2] != 1 || w->ne[3] != 1)
        throw std::runtime_error("CLAP: linear shape: " + name);
    const int64_t rows = x->ne[1], columns = weights.weight->ne[1];
    // CPU-decoded F32 weights need the same Vulkan tile padding as uploaded weights.
    if (runtime.vulkan && w->ne[1] % 32)
        w = ggml_pad(ctx(), w, 0, (w->ne[1] + 31) / 32 * 32 - w->ne[1], 0, 0);
    if (runtime.vulkan && rows > 8 && rows % 32)
        x = ggml_pad(ctx(), x, 0, (rows + 31) / 32 * 32 - rows, 0, 0);
    auto *y = matmul(w, x, true);
    if (y->ne[0] != columns || y->ne[1] != rows)
        y = ggml_cont(ctx(), ggml_view_2d(ctx(), y, columns, rows, y->nb[1], 0));
    if (weights.bias)
        y = ggml_add(ctx(), y, weight(weights.bias));
    return y;
}
ggml_tensor *Graph::norm(ggml_tensor *x, const LayerWeights &weights, float epsilon) {
    return ggml_add(ctx(), ggml_mul(ctx(), ggml_norm(ctx(), x, epsilon), weight(weights.weight)), weight(weights.bias));
}
ggml_tensor *Graph::embedding(ggml_tensor *source, const std::vector<int> &rows) {
    auto *w = weight(source, false);
    for (int row : rows)
        if (row < 0 || row >= w->ne[1])
            throw std::runtime_error("CLAP: embedding index out of range");
    return ggml_get_rows(ctx(), w, indices(rows));
}
ggml_tensor *Graph::embedding(ggml_tensor *source, ggml_tensor *rows) {
    return ggml_get_rows(ctx(), weight(source, false), rows);
}
ggml_tensor *Graph::weight(const std::string &name, bool f32) { return weight(runtime.loader.weight(name), f32); }
ggml_tensor *Graph::linear(ggml_tensor *x, const std::string &name, bool bias) {
    return linear(x, LayerWeights::bind(runtime.loader, name, bias));
}
ggml_tensor *Graph::norm(ggml_tensor *x, const std::string &name, float epsilon) {
    return norm(x, LayerWeights::bind(runtime.loader, name), epsilon);
}
ggml_tensor *Graph::conv1d(ggml_tensor *x, const std::string &name, int kernel, int stride, int padding, int groups,
                           bool bias) {
    return conv1d(x, LayerWeights::bind(runtime.loader, name, bias), kernel, stride, padding, groups);
}
ggml_tensor *Graph::embedding(const std::string &name, const std::vector<int> &rows) {
    return embedding(runtime.loader.weight(name), rows);
}
ggml_tensor *Graph::slice(ggml_tensor *x, int64_t first, int64_t columns) {
    return ggml_cont(ctx(), ggml_view_2d(ctx(), x, columns, x->ne[1], x->nb[1], first * x->nb[0]));
}
ggml_tensor *Graph::conv1d(ggml_tensor *x, const LayerWeights &weights, int kernel, int stride, int padding,
                           int groups) {
    const std::string name = ggml_get_name(weights.weight);
    auto *w = weight(weights.weight);
    if (kernel <= 0 || stride <= 0 || padding < 0 || groups <= 0 || w->ne[0] != kernel ||
        w->ne[1] * groups != x->ne[0] || w->ne[2] % groups || w->ne[3] != 1)
        throw std::runtime_error("CLAP: convolution shape: " + name);
    const int64_t rows = (x->ne[1] + 2 * padding - kernel) / stride + 1;
    if (rows <= 0)
        throw std::runtime_error("CLAP: convolution input is too short: " + name);
    const int64_t inputs = w->ne[1], outputs = w->ne[2] / groups;
    ggml_tensor *result = nullptr;
    // Batch depthwise channels as independent images instead of one graph branch per channel.
    if (groups == x->ne[0] && outputs == 1) {
        auto *data = ggml_reshape_3d(ctx(), ggml_cont(ctx(), ggml_transpose(ctx(), x)), x->ne[1], 1, groups);
        auto *patches = ggml_im2col(ctx(), w, data, stride, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
        auto *kernels = ggml_reshape_3d(ctx(), w, kernel, 1, groups);
        auto *y = matmul(kernels, patches);
        result = ggml_cont(ctx(), ggml_transpose(ctx(), ggml_reshape_2d(ctx(), y, rows, groups)));
    } else {
        for (int group = 0; group < groups; ++group) {
            auto *data = ggml_cont(ctx(), ggml_transpose(ctx(), slice(x, group * inputs, inputs)));
            auto *kernels =
                ggml_view_3d(ctx(), w, kernel, inputs, outputs, w->nb[1], w->nb[2], size_t(group) * outputs * w->nb[2]);
            auto *patches = ggml_im2col(ctx(), kernels, data, stride, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
            auto *y = matmul(ggml_reshape_2d(ctx(), ggml_cont(ctx(), kernels), kernel * inputs, outputs), patches);
            result = result ? ggml_concat(ctx(), result, y, 0) : y;
        }
    }
    if (weights.bias)
        result = ggml_add(ctx(), result, weight(weights.bias));
    return result;
}
ggml_tensor *Graph::mean_pool(ggml_tensor *x, const std::vector<int> &mask) {
    if (mask.empty())
        return ggml_cont(ctx(), ggml_transpose(ctx(), ggml_mean(ctx(), ggml_cont(ctx(), ggml_transpose(ctx(), x)))));
    std::vector<float> values(mask.size());
    int count = 0;
    for (size_t i = 0; i < mask.size(); ++i) {
        values[i] = mask[i] ? 1.0f : 0.0f;
        count += mask[i] != 0;
    }
    if (!count || int64_t(mask.size()) != x->ne[1])
        throw std::runtime_error("CLAP: invalid pooling mask");
    auto *weighted = ggml_mul(ctx(), x, input(values, 1, values.size()));
    return ggml_scale(
        ctx(),
        ggml_cont(ctx(),
                  ggml_transpose(ctx(), ggml_sum_rows(ctx(), ggml_cont(ctx(), ggml_transpose(ctx(), weighted))))),
        1.0f / count);
}
ggml_tensor *Graph::l2_normalize(ggml_tensor *x, float epsilon) { return ggml_l2_norm(ctx(), x, epsilon); }
ggml_tensor *Graph::attention(ggml_tensor *q, ggml_tensor *k, ggml_tensor *v, int heads, ggml_tensor *mask, float scale,
                              int window_tokens) {
    const int64_t width = q->ne[0], rows = q->ne[1], tokens = window_tokens ? window_tokens : rows, dim = width / heads,
                  windows = rows / tokens;
    q = ggml_permute(ctx(), ggml_reshape_4d(ctx(), q, dim, heads, tokens, windows), 0, 2, 1, 3);
    k = ggml_permute(ctx(), ggml_reshape_4d(ctx(), k, dim, heads, tokens, windows), 0, 2, 1, 3);
    v = ggml_permute(ctx(), ggml_reshape_4d(ctx(), v, dim, heads, tokens, windows), 1, 2, 0, 3);
    const int64_t padded_tokens = runtime.vulkan && tokens > 8 ? (tokens + 31) / 32 * 32 : tokens;
    const int64_t padded_dim = runtime.vulkan && tokens > 8 ? (dim + 31) / 32 * 32 : dim;
    if (padded_tokens != tokens || padded_dim != dim) {
        q = ggml_pad(ctx(), q, 0, int(padded_tokens - tokens), 0, 0);
        k = ggml_pad(ctx(), k, 0, int(padded_tokens - tokens), 0, 0);
        v = ggml_pad(ctx(), v, int(padded_tokens - tokens), int(padded_dim - dim), 0, 0);
        if (mask)
            mask = ggml_pad(ctx(), mask, int(padded_tokens - tokens),
                            mask->ne[1] == tokens ? int(padded_tokens - tokens) : 0, 0, 0);
    }
    auto *scores = matmul(k, q);
    scores = ggml_scale(ctx(), scores, scale == 0 ? 1.0f / std::sqrt(float(dim)) : scale);
    if (mask)
        scores = ggml_add(ctx(), scores, mask);
    if (padded_tokens != tokens) {
        std::vector<float> padding_mask(padded_tokens);
        std::fill(padding_mask.begin() + tokens, padding_mask.end(), -INFINITY);
        scores = ggml_add(ctx(), scores, input(padding_mask, padded_tokens, 1));
    }
    auto *out = matmul(ggml_cont(ctx(), v), ggml_soft_max(ctx(), scores));
    if (padded_tokens != tokens || padded_dim != dim)
        out = ggml_view_4d(ctx(), out, dim, tokens, heads, windows, out->nb[1], out->nb[2], out->nb[3], 0);
    return ggml_reshape_2d(ctx(), ggml_cont(ctx(), ggml_permute(ctx(), out, 0, 2, 1, 3)), width, rows);
}
ggml_tensor *Graph::rope(ggml_tensor *x, int heads, float base) {
    const int64_t width = x->ne[0], tokens = x->ne[1], dim = width / heads;
    std::vector<int> positions(tokens);
    for (int64_t i = 0; i < tokens; ++i)
        positions[i] = int(i);
    auto *out = ggml_rope_ext(ctx(), ggml_reshape_3d(ctx(), x, dim, heads, tokens), indices(positions), nullptr,
                              int(dim), GGML_ROPE_TYPE_NEOX, 0, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_reshape_2d(ctx(), out, width, tokens);
}
void Graph::compute(const std::vector<ggml_tensor *> &outputs) {
    allocate(outputs);
    execute();
}
void Graph::allocate(const std::vector<ggml_tensor *> &outputs) {
    graph = ggml_new_graph_custom(ctx(), capacity, false);
    for (auto *output : outputs) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
    const size_t needed = size_t(ggml_graph_n_nodes(graph)) * 4 + runtime.loader.tensor_count() * 2 + 128;
    if (needed > runtime.scheduler_capacity) {
        runtime.scheduler.reset();
        ggml_backend_t backends[] = {runtime.gpu.get(), runtime.cpu.get()};
        runtime.scheduler.reset(ggml_backend_sched_new(backends + (runtime.gpu ? 0 : 1), nullptr, runtime.gpu ? 2 : 1,
                                                       needed, false, true));
        if (!runtime.scheduler)
            throw std::runtime_error("CLAP: scheduler allocation failed");
        runtime.scheduler_capacity = needed;
    }
    for (auto *node : cpu_nodes)
        ggml_backend_sched_set_tensor_backend(runtime.scheduler.get(), node, runtime.cpu.get());
    for (auto *node : device_nodes)
        ggml_backend_sched_set_tensor_backend(runtime.scheduler.get(), node, runtime.gpu.get());
    if (!ggml_backend_sched_alloc_graph(runtime.scheduler.get(), graph))
        throw std::runtime_error("CLAP: graph allocation failed");
}
void Graph::execute() {
    // The allocator can reuse input storage for later intermediates. Restore
    // constants/indices as well as caller inputs on every execution.
    for (const auto &input : inputs)
        if (input.tensor->buffer)
            ggml_backend_tensor_set(input.tensor, input.bytes.data(), 0, input.bytes.size());

    if (ggml_backend_sched_graph_compute(runtime.scheduler.get(), graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("CLAP: graph execution failed");
    runtime.last_gpu = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto *node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT && runtime.gpu &&
            ggml_backend_sched_get_tensor_backend(runtime.scheduler.get(), node) == runtime.gpu.get())
            runtime.last_gpu = true;
    }
}
std::vector<float> Graph::read(ggml_tensor *output) const {
    if (output->type != GGML_TYPE_F32 || !ggml_is_contiguous(output))
        throw std::runtime_error("CLAP: output must be contiguous F32");
    std::vector<float> values(ggml_nelements(output));
    ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
    return values;
}
std::vector<float> Graph::run(ggml_tensor *output) {
    output = ggml_cont(ctx(), output);
    compute({output});
    return read(output);
}
} // namespace clap::detail
