#pragma once

#include "model-loader.h"

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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

struct LayerWeights {
    ggml_tensor *weight = nullptr;
    ggml_tensor *bias = nullptr;
    static LayerWeights bind(const ModelLoader &loader, const std::string &name, bool bias = true) {
        return {loader.weight(name + ".weight"), bias ? loader.weight(name + ".bias") : nullptr};
    }
};

// Borrowed host data; run_cached() completes all uploads before returning.
struct GraphInput {
    enum ggml_type type;
    int64_t columns, rows;
    const void *data;
    size_t bytes;
    GraphInput(const std::vector<float> &values, int64_t columns, int64_t rows);
    explicit GraphInput(const HostMatrix &values) : GraphInput(values.data, values.cols, values.rows) {}
    explicit GraphInput(const std::vector<int> &values);
};
struct GraphOutput {
    std::vector<float> values;
    int64_t columns, rows;
};
struct GraphStats {
    size_t builds = 0, hits = 0, weight_uploads = 0;
    std::string device;
    bool device_matmul = false;
};
class Graph;

// Persistent backend and immutable weight ownership. One graph executes at a time.
class GraphRuntime {
  public:
    GraphRuntime(const ModelLoader &loader, int threads, bool f32_activations = false);
    ~GraphRuntime();
    bool last_graph_used_vulkan() const;
    size_t vulkan_weight_uploads() const;
    GraphStats stats() const;
    // One bounded cache per runtime. The key encodes branch and scalar options;
    // input types/shapes are also checked. Builders run only on cache misses.
    std::vector<GraphOutput> run_cached(const std::string &key, const std::vector<GraphInput> &inputs,
                                        const std::function<std::vector<ggml_tensor *>(Graph &)> &build,
                                        size_t extra_nodes = 0) const;

  private:
    friend class Graph;
    struct State;
    std::unique_ptr<State> state;
};

// Graph builders create nodes only; compute() allocates and executes the graph.
class Graph {
  public:
    explicit Graph(const GraphRuntime &runtime, size_t extra_nodes = 0);
    ~Graph();
    Graph(const Graph &) = delete;
    Graph &operator=(const Graph &) = delete;
    ggml_context *ctx() const { return context.get(); }
    ggml_tensor *input(const std::vector<float> &data, int64_t columns, int64_t rows);
    ggml_tensor *input(const HostMatrix &data) { return input(data.data, data.cols, data.rows); }
    ggml_tensor *indices(const std::vector<int> &data);
    ggml_tensor *dynamic_input(size_t index) const;
    ggml_tensor *weight(ggml_tensor *source, bool f32 = true);
    ggml_tensor *matmul(ggml_tensor *weight, ggml_tensor *input, bool offload = false);
    ggml_tensor *linear(ggml_tensor *x, const LayerWeights &weights);
    ggml_tensor *norm(ggml_tensor *x, const LayerWeights &weights, float epsilon);
    ggml_tensor *conv1d(ggml_tensor *x, const LayerWeights &weights, int kernel, int stride, int padding, int groups);
    ggml_tensor *embedding(ggml_tensor *weight, ggml_tensor *indices);
    ggml_tensor *embedding(ggml_tensor *weight, const std::vector<int> &indices);
    ggml_tensor *weight(const std::string &name, bool f32 = true);
    ggml_tensor *linear(ggml_tensor *x, const std::string &name, bool bias = true);
    ggml_tensor *norm(ggml_tensor *x, const std::string &name, float epsilon);
    ggml_tensor *conv1d(ggml_tensor *x, const std::string &name, int kernel, int stride, int padding, int groups,
                        bool bias = true);
    ggml_tensor *embedding(const std::string &name, const std::vector<int> &rows);
    ggml_tensor *slice(ggml_tensor *x, int64_t first, int64_t columns);
    ggml_tensor *mean_pool(ggml_tensor *x, const std::vector<int> &mask = {});
    ggml_tensor *l2_normalize(ggml_tensor *x, float epsilon = 1e-12f);
    ggml_tensor *attention(ggml_tensor *q, ggml_tensor *k, ggml_tensor *v, int heads, ggml_tensor *mask = nullptr,
                           float scale = 0, int window_tokens = 0);
    ggml_tensor *rope(ggml_tensor *x, int heads, float base);
    void compute(const std::vector<ggml_tensor *> &outputs);
    std::vector<float> read(ggml_tensor *output) const;
    std::vector<float> run(ggml_tensor *output);

  private:
    friend class GraphRuntime;
    Graph(const GraphRuntime &runtime, size_t extra_nodes, bool cached);
    void allocate(const std::vector<ggml_tensor *> &outputs);
    void execute();
    void update(const std::vector<GraphInput> &inputs);
    GraphRuntime::State &runtime;
    std::unique_lock<std::mutex> lock;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    struct Input {
        ggml_tensor *tensor;
        std::vector<uint8_t> bytes;
    };
    std::vector<Input> inputs;
    std::unordered_map<ggml_tensor *, ggml_tensor *> converted;
    std::vector<ggml_tensor *> cpu_nodes, device_nodes;
    std::vector<ggml_tensor *> dynamic_inputs;
    std::vector<ggml_tensor *> outputs;
    ggml_cgraph *graph = nullptr;
    size_t capacity;
    ggml_tensor *raw_weight(ggml_tensor *source, bool dense);
};
} // namespace clap::detail
