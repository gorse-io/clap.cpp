#include "graph.h"
#include <gguf.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace clap::detail;
static void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
static void close(float actual, float expected, float tolerance = 3e-6f) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error("graph differs from scalar oracle: " + std::to_string(actual) + " vs " +
                                 std::to_string(expected));
}
static void convolution(const GraphRuntime &runtime, const ModelLoader &loader, int groups, const std::string &name) {
    HostMatrix input(11, 4);
    for (size_t i = 0; i < input.data.size(); ++i)
        input.data[i] = std::sin(float(i) * 0.31f);
    const auto *weight = loader.weight(name + ".weight"), *bias = loader.weight(name + ".bias");
    const int columns = int(weight->ne[2]), per_group = columns / groups, inputs = input.cols / groups;
    Graph graph(runtime);
    auto *ctx = graph.ctx();
    auto *output = graph.conv1d(graph.input(input), name, 3, 2, 1, groups);
    output = ggml_add(ctx, ggml_gelu_erf(ctx, output), output);
    const auto values = graph.run(output);
    for (int row = 0; row < output->ne[1]; ++row)
        for (int column = 0; column < columns; ++column) {
            float expected = scalar(bias, column);
            for (int channel = 0; channel < inputs; ++channel)
                for (int tap = 0; tap < 3; ++tap) {
                    const int source = row * 2 + tap - 1;
                    if (source >= 0 && source < input.rows)
                        expected += input[source][column / per_group * inputs + channel] *
                                    scalar(weight, (column * inputs + channel) * 3 + tap);
                }
            const float activated = 0.5f * expected * (1.0f + std::erf(expected / std::sqrt(2.0f)));
            close(values[row * columns + column], expected + activated);
        }
}
template <int tokens> static void attention(const GraphRuntime &runtime) {
    constexpr int rows = 2 * tokens, columns = 8, heads = 2, dim = columns / heads;
    HostMatrix q(rows, columns), k(rows, columns), v(rows, columns);
    for (size_t i = 0; i < q.data.size(); ++i) {
        q.data[i] = std::sin(float(i) * 0.17f);
        k.data[i] = std::cos(float(i) * 0.23f);
        v.data[i] = std::sin(float(i) * 0.41f);
    }
    Graph graph(runtime);
    std::vector<float> mask(tokens * tokens * 2);
    for (int window = 0; window < 2; ++window)
        for (int row = 0; row < tokens; ++row)
            mask[(window * tokens + row) * tokens + window] = -1e30f;
    auto *bias = ggml_reshape_4d(graph.ctx(), graph.input(mask, tokens, tokens * 2), tokens, tokens, 1, 2);
    auto *output = graph.attention(graph.input(q), graph.input(k), graph.input(v), heads, bias, 0, tokens);
    const auto actual = graph.run(output);
    for (int window = 0; window < 2; ++window)
        for (int row = 0; row < tokens; ++row)
            for (int head = 0; head < heads; ++head) {
                float scores[tokens], maximum = -INFINITY;
                for (int source = 0; source < tokens; ++source) {
                    float score = 0;
                    for (int d = 0; d < dim; ++d)
                        score += q[window * tokens + row][head * dim + d] * k[window * tokens + source][head * dim + d];
                    scores[source] = score / std::sqrt(float(dim)) + mask[(window * tokens + row) * tokens + source];
                    maximum = std::max(maximum, scores[source]);
                }
                float sum = 0;
                for (float &score : scores) {
                    score = std::exp(score - maximum);
                    sum += score;
                }
                for (int d = 0; d < dim; ++d) {
                    float expected = 0;
                    for (int source = 0; source < tokens; ++source)
                        expected += scores[source] / sum * v[window * tokens + source][head * dim + d];
                    close(actual[(window * tokens + row) * columns + head * dim + d], expected);
                }
            }
}
static void normalization(const GraphRuntime &runtime) {
    HostMatrix input(9, 8);
    for (size_t i = 0; i < input.data.size(); ++i)
        input.data[i] = std::cos(float(i) * 0.13f);
    Graph graph(runtime);
    auto *rotated = graph.rope(graph.input(input), 2, 10000.0f);
    auto *pooled = graph.mean_pool(rotated, {1, 0, 1, 1, 0, 0, 1, 0, 1});
    graph.compute({rotated, pooled});
    const auto actual = graph.read(rotated), mean = graph.read(pooled);
    std::vector<float> expected_mean(8);
    for (int row = 0; row < input.rows; ++row)
        for (int head = 0; head < 2; ++head)
            for (int d = 0; d < 2; ++d) {
                const float angle = row * std::pow(10000.0f, -2.0f * d / 4);
                const int c = head * 4 + d;
                close(actual[row * 8 + c], input[row][c] * std::cos(angle) - input[row][c + 2] * std::sin(angle));
                close(actual[row * 8 + c + 2], input[row][c + 2] * std::cos(angle) + input[row][c] * std::sin(angle));
            }
    for (int row : {0, 2, 3, 6, 8})
        for (int c = 0; c < 8; ++c)
            expected_mean[c] += actual[row * 8 + c] / 5;
    for (int c = 0; c < 8; ++c)
        close(mean[c], expected_mean[c]);
}
static void cache(const GraphRuntime &runtime) {
    const auto before = runtime.stats();
    std::vector<float> values{1, 2, 3, 4, 5, 6};
    std::vector<int> indices{2, 0};
    int builds = 0;
    auto build = [&](Graph &g) {
        ++builds;
        auto *selected = ggml_get_rows(g.ctx(), g.dynamic_input(0), g.dynamic_input(1));
        // A static input must also survive reuse and subsequent cache eviction.
        auto *result = ggml_add(g.ctx(), selected, g.input({0.25f}, 1, 1));
        return std::vector<ggml_tensor *>{result, g.mean_pool(result)};
    };
    auto run = [&](const std::string &key) {
        const auto result =
            runtime.run_cached(key, {GraphInput(values, 2, values.size() / 2), GraphInput(indices)}, build);
        require(result.size() == 2 && result[0].columns == 2 && result[0].rows == int64_t(indices.size()),
                "cached output shape");
        for (size_t row = 0; row < indices.size(); ++row)
            for (int column = 0; column < 2; ++column)
                close(result[0].values[row * 2 + column], values[indices[row] * 2 + column] + 0.25f);
        for (int column = 0; column < 2; ++column) {
            float mean = 0;
            for (int row : indices)
                mean += (values[row * 2 + column] + 0.25f) / indices.size();
            close(result[1].values[column], mean);
        }
    };
    run("a");
    for (float &value : values)
        value *= -2;
    indices = {1, 2};
    run("a");
    require(builds == 1, "same-shape graph was rebuilt");
    values.insert(values.end(), {7, 8});
    indices = {3, 1};
    run("a");
    require(builds == 2, "shape change failed to invalidate graph");
    run("b");
    require(builds == 3, "branch change failed to invalidate graph");
    bool failed = false;
    try {
        runtime.run_cached("failure", {GraphInput(values, 2, 4)}, [](Graph &) -> std::vector<ggml_tensor *> {
            throw std::runtime_error("intentional builder failure");
        });
    } catch (const std::runtime_error &) {
        failed = true;
    }
    require(failed, "builder failure did not propagate");
    run("b");
    require(builds == 4, "graph not rebuilt after failure");
    const auto after = runtime.stats();
    require(after.builds - before.builds == 4 && after.hits - before.hits == 1, "cache accounting");
}
static void precision(const std::string &path) {
    ModelLoader loader(clap::RuntimeOptions{2, {}, clap::ActivationPrecision::F32});
    loader.load(path);
    GraphRuntime runtime(loader, 2);
    const auto weights = LayerWeights::bind(loader, "dense");
    HostMatrix input(11, 64);
    const auto device_name = runtime.stats().device;
    auto device = ggml_backend_dev_by_name(device_name.c_str());
    bool supports_matmul = false;
    auto build = [&](Graph &g) {
        auto *output = g.linear(g.dynamic_input(0), weights);
        // A device can initialize without supporting this graph's F32 matmul
        // (for example, Metal on Apple's paravirtual CI GPU). Keep checking
        // numerical correctness on CPU, but require offload on capable GPUs.
        auto *graph = ggml_new_graph(g.ctx());
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto *node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT && device_name != "CPU" && ggml_backend_dev_supports_op(device, node))
                supports_matmul = true;
        }
        return std::vector<ggml_tensor *>{output};
    };
    size_t uploads = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < input.data.size(); ++i)
            input.data[i] = std::sin(float(i + pass) * 0.173f);
        const auto result = runtime.run_cached("dense", {GraphInput(input)}, build);
        for (int row = 0; row < input.rows; ++row)
            for (int column = 0; column < 16; ++column) {
                float expected = scalar(weights.bias, column);
                for (int i = 0; i < input.cols; ++i)
                    expected += input[row][i] * scalar(weights.weight, column * input.cols + i);
                close(result[0].values[row * 16 + column], expected, 2e-5f);
            }
        const auto stats = runtime.stats();
        if (pass == 0)
            uploads = stats.weight_uploads;
        else
            require(stats.builds == 1 && stats.hits == 1 && stats.weight_uploads == uploads,
                    "dense graph/weights not reused");
        require(stats.device_matmul == supports_matmul, "F32 matmul placement differs from device capability");
    }
}
static void devices(const std::string &path) {
    const auto devices = clap::available_devices();
    require(!devices.empty(), "GGML device enumeration is empty");
    ModelLoader cpu(clap::RuntimeOptions{2, "cpu", clap::ActivationPrecision::F32});
    cpu.load(path);
    GraphRuntime runtime(cpu, 2);
    cache(runtime);
    ModelLoader invalid(clap::RuntimeOptions{2, "missing-device-for-test"});
    invalid.load(path);
    bool rejected = false;
    try {
        GraphRuntime bad(invalid, 2);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "explicit unavailable device must fail");
}
int main(int argc, char **argv) {
    try {
        require(argc == 2, "graph-pipeline-test FIXTURE.gguf");
        auto *ctx = ggml_init({1024 * 1024, nullptr, false});
        auto *file = gguf_init_empty();
        for (int groups : {1, 2, 4}) {
            const std::string name = "conv" + std::to_string(groups);
            auto *weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4 / groups, groups == 2 ? 6 : 4);
            auto *bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, weight->ne[2]);
            for (int64_t i = 0; i < ggml_nelements(weight); ++i)
                static_cast<float *>(weight->data)[i] = std::cos(float(i) * 0.43f) * 0.17f;
            for (int64_t i = 0; i < ggml_nelements(bias); ++i)
                static_cast<float *>(bias->data)[i] = float(i) * 0.07f;
            ggml_set_name(weight, (name + ".weight").c_str());
            ggml_set_name(bias, (name + ".bias").c_str());
            gguf_add_tensor(file, weight);
            gguf_add_tensor(file, bias);
        }
        auto *dense = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 16);
        auto *bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        for (int i = 0; i < 1024; ++i)
            static_cast<ggml_fp16_t *>(dense->data)[i] = ggml_fp32_to_fp16(std::cos(float(i) * 0.37f) * 0.2f);
        for (int i = 0; i < 16; ++i)
            static_cast<float *>(bias->data)[i] = i * 0.01f;
        ggml_set_name(dense, "dense.weight");
        ggml_set_name(bias, "dense.bias");
        gguf_add_tensor(file, dense);
        gguf_add_tensor(file, bias);
        require(gguf_write_to_file(file, argv[1], false), "cannot write graph fixture");
        gguf_free(file);
        ggml_free(ctx);
        ModelLoader loader;
        loader.load(argv[1]);
        GraphRuntime runtime(loader, 2);
        for (int groups : {1, 2, 4})
            convolution(runtime, loader, groups, "conv" + std::to_string(groups));
        attention<3>(runtime);
        attention<9>(runtime);
        attention<17>(runtime);
        normalization(runtime);
        cache(runtime);
        devices(argv[1]);
        precision(argv[1]);
        std::filesystem::remove(argv[1]);
        std::cout << "GGML operators, device selection, activation precision and graph reuse passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
