#include "model-ops.h"
#include <gguf.h>
#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using clap::detail::Matrix;
using clap::detail::ModelLoader;
using clap::detail::TensorOps;

namespace {
std::filesystem::path fixture_path;
bool require_vulkan = false;
bool require_cpu = false;

class TensorOpsTest : public testing::Test {
protected:
    void TearDown() override {
        std::error_code error;
        std::filesystem::remove(fixture_path, error);
        EXPECT_FALSE(error) << "fixture cleanup failed: " << error.message();
    }
};

TEST_F(TensorOpsTest, LinearMatchesScalarAcrossTypesShapesAndRepeatedCalls) {
    const auto &path = fixture_path;
    const bool strict = require_vulkan;
    const bool cpu = require_cpu;
    const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0,
                               GGML_TYPE_Q4_K, GGML_TYPE_Q6_K};
    const int output_sizes[] = {31, 32, 33, 37, 41, 64, 65};
    const int batch_sizes[] = {1, 9, 1, 9, 7, 8, 17, 33, 65};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({1024 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> fixture(gguf_init_empty(), gguf_free);
    auto *ctx = context.get();
    auto *file = fixture.get();
    ASSERT_NE(nullptr, ctx) << "fixture allocation failed";
    ASSERT_NE(nullptr, file) << "fixture allocation failed";
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
    fixture.reset();
    context.reset();
    ASSERT_TRUE(written) << "cannot write fixture";
    ModelLoader loader;
    loader.load(path.string());
    TensorOps ops(loader, 2);
    size_t expected_uploads = 0;
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
            SCOPED_TRACE(testing::Message() << "type=" << name << " call=" << call
                         << " shape=" << x.rows << 'x' << x.cols << " bias=" << bias);
            ASSERT_EQ(x.rows, result.rows) << "linear returned padded dimensions";
            ASSERT_EQ(out, result.cols) << "linear returned padded dimensions";
            const auto *w = loader.weight(name + ".weight");
            const auto *b = loader.weight(name + ".bias");
            for (int r = 0; r < x.rows; ++r) for (int c = 0; c < out; ++c) {
                double expected = bias ? clap::detail::scalar(b, c) : 0;
                for (int k = 0; k < in; ++k)
                    expected += double(x[r][k]) * clap::detail::scalar(w, size_t(c) * in + k);
                EXPECT_TRUE(std::isfinite(result[r][c])) << "row=" << r << " col=" << c;
                EXPECT_LT(std::abs(result[r][c] - expected), 0.002)
                    << "row=" << r << " col=" << c << " actual=" << result[r][c]
                    << " expected=" << expected << " vulkan=" << ops.last_linear_used_vulkan();
            }
            if (cpu) {
                EXPECT_FALSE(ops.last_linear_used_vulkan()) << "CPU fallback was ignored";
                EXPECT_EQ(0u, ops.vulkan_weight_uploads()) << "CPU path uploaded weights to Vulkan";
            }
            if (strict) {
                EXPECT_TRUE(ops.last_linear_used_vulkan()) << "strict mode: linear did not execute on Vulkan";
                if (call == 0) ++expected_uploads;
                EXPECT_EQ(expected_uploads, ops.vulkan_weight_uploads())
                    << "weights were not cached across repeated calls";
            }
            ++call;
        }
    }
}
} // namespace

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--require-vulkan" && std::string(argv[2]) != "--require-cpu")) {
        std::cerr << "usage: tensor-ops-test FIXTURE.gguf [--require-vulkan|--require-cpu]\n";
        return 2;
    }
    fixture_path = argv[1];
    require_vulkan = argc == 3 && std::string(argv[2]) == "--require-vulkan";
    require_cpu = argc == 3 && std::string(argv[2]) == "--require-cpu";
    return RUN_ALL_TESTS();
}
