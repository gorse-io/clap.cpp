#include "model-ops.h"
#include <gguf.h>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using clap::detail::Matrix;
using clap::detail::ModelLoader;
using clap::detail::TensorOps;

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void run(const std::filesystem::path &path, bool strict, bool cpu) {
    const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0,
                               GGML_TYPE_Q4_K, GGML_TYPE_Q6_K};
    const int output_sizes[] = {31, 32, 33, 37, 41, 64, 65};
    const int batch_sizes[] = {1, 9, 1, 9, 7, 8, 17, 33, 65};
    auto *ctx = ggml_init({1024 * 1024, nullptr, false});
    auto *file = gguf_init_empty();
    require(ctx && file, "fixture allocation failed");
    for (int out : output_sizes) for (auto type : types) {
        const bool quantized = ggml_is_quantized(type);
        const int in = quantized ? 256 : 64;
        const std::string name = std::string(ggml_type_name(type)) + "." + std::to_string(out);
        auto *w = ggml_new_tensor_2d(ctx, type, in, out);
        auto *b = ggml_new_tensor_1d(ctx, quantized ? GGML_TYPE_F32 : type, out);
        ggml_set_name(w, (name + ".weight").c_str());
        ggml_set_name(b, (name + ".bias").c_str());
        std::vector<float> values(in * out);
        for (int j = 0; j < in * out; ++j) {
            const float v = std::sin(float(j) * 0.17f) * 0.125f;
            values[j] = v;
            if (type == GGML_TYPE_F32) static_cast<float *>(w->data)[j] = v;
            else if (type == GGML_TYPE_F16) static_cast<ggml_fp16_t *>(w->data)[j] = ggml_fp32_to_fp16(v);
        }
        if (quantized) ggml_get_type_traits(type)->from_float_ref(values.data(), w->data, values.size());
        for (int j = 0; j < out; ++j) {
            const float v = float(j - 15) * 0.03125f;
            if (b->type == GGML_TYPE_F32) static_cast<float *>(b->data)[j] = v;
            else static_cast<ggml_fp16_t *>(b->data)[j] = ggml_fp32_to_fp16(v);
        }
        gguf_add_tensor(file, w);
        gguf_add_tensor(file, b);
    }
    const bool written = gguf_write_to_file(file, path.string().c_str(), false);
    gguf_free(file);
    ggml_free(ctx);
    require(written, "cannot write fixture");
    ModelLoader loader;
    loader.load(path.string());
    TensorOps ops(loader, 2);
    size_t expected_uploads = 0;
    bool parity = true;
    for (int out : output_sizes) for (auto type : types) {
        const bool quantized = ggml_is_quantized(type);
        const int in = quantized ? 256 : 64;
        const std::string name = std::string(ggml_type_name(type)) + "." + std::to_string(out);
        int call = 0;
        for (int rows : batch_sizes) {
            Matrix x(rows, in);
            for (size_t i = 0; i < x.data.size(); ++i)
                // Exactly representable constant rows isolate weight decoding and
                // tile padding from CPU activation-packing approximation.
                x.data[i] = quantized ? (int(i / in) % 3 - 1) * 0.25f :
                    std::cos(float(i) * 0.13f + float(call)) * 0.5f;
            const bool bias = call % 2 == 0;
            const auto result = ops.linear(x, name, bias);
            require(result.rows == x.rows && result.cols == out, "linear returned padded dimensions");
            const auto *w = loader.weight(name + ".weight");
            const auto *b = loader.weight(name + ".bias");
            for (int r = 0; r < x.rows; ++r) for (int c = 0; c < out; ++c) {
                double expected = bias ? clap::detail::scalar(b, c) : 0;
                for (int k = 0; k < in; ++k)
                    expected += double(x[r][k]) * clap::detail::scalar(w, size_t(c) * in + k);
                if (!std::isfinite(result[r][c]) || std::abs(result[r][c] - expected) >= 0.002) {
                    std::cerr << "linear mismatch: type=" << name << " call=" << call
                              << " shape=" << x.rows << 'x' << x.cols << " bias=" << bias
                              << " row=" << r << " col=" << c << " actual=" << result[r][c]
                              << " expected=" << expected << " error=" << result[r][c] - expected
                              << " vulkan=" << ops.last_linear_used_vulkan() << '\n';
                    parity = false;
                }
            }
            if (cpu) {
                require(!ops.last_linear_used_vulkan(), "CPU fallback was ignored");
                require(ops.vulkan_weight_uploads() == 0, "CPU path uploaded weights to Vulkan");
            }
            if (strict) {
                require(ops.last_linear_used_vulkan(), "strict mode: linear did not execute on Vulkan");
                if (call == 0) ++expected_uploads;
                require(ops.vulkan_weight_uploads() == expected_uploads, "weights were not cached across repeated calls");
            }
            ++call;
        }
    }
    require(parity, "linear disagrees with scalar oracle");
    std::cout << "F32/F16/Q8_0/Q4_K/Q6_K scalar parity, bias/no-bias, repeated calls: PASS ("
              << (ops.last_linear_used_vulkan() ? "Vulkan" : "CPU") << ")\n";
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--require-vulkan" && std::string(argv[2]) != "--require-cpu")) {
        std::cerr << "usage: tensor-ops-test FIXTURE.gguf [--require-vulkan|--require-cpu]\n";
        return 2;
    }
    const std::filesystem::path path(argv[1]);
    try {
        run(path, argc == 3 && std::string(argv[2]) == "--require-vulkan",
            argc == 3 && std::string(argv[2]) == "--require-cpu");
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception &e) {
        std::filesystem::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
