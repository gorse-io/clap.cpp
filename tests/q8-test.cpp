#include "model-loader.h"
#include "model-ops.h"
#include <ggml.h>
#include <gguf.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace clap::detail;
static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void create(const std::string &path, ggml_type type, uint32_t alignment = 32) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({8 * 1024 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    gguf_set_val_str(file.get(), "general.architecture", "clap");
    gguf_set_val_str(file.get(), "test.metadata", "preserved");
    const int32_t metadata[] = {7, -3, 42};
    gguf_set_arr_data(file.get(), "test.array", GGUF_TYPE_INT32, metadata, 3);
    // Reload the empty header: gguf_set_val_u32 alone does not change the
    // writer context's internal alignment. This produces a valid 64-byte fixture.
    gguf_set_val_u32(file.get(), "general.alignment", alignment);
    require(gguf_write_to_file(file.get(), path.c_str(), false), "fixture header write failed");
    file.reset(gguf_init_from_file(path.c_str(), {true, nullptr}));
    require(file && gguf_get_alignment(file.get()) == alignment, "fixture alignment not applied");
    auto add = [&](const char *name, std::vector<int64_t> shape, bool quantizable) {
        const auto storage_type = quantizable ? type :
            (shape.size() > 1 && type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
        auto *t = ggml_new_tensor(ctx.get(), storage_type, static_cast<int>(shape.size()), shape.data());
        ggml_set_name(t, name);
        std::vector<float> data(ggml_nelements(t));
        for (size_t i = 32; i < data.size(); ++i) data[i] = std::sin(static_cast<float>(i) * 0.13f) * 0.1f;
        if (t->type == GGML_TYPE_F32) std::memcpy(t->data, data.data(), ggml_nbytes(t));
        else if (t->type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(data.data(), static_cast<ggml_fp16_t *>(t->data), data.size());
        else ggml_quantize_chunk(type, data.data(), t->data, 0, ggml_nrows(t), t->ne[0], nullptr);
        gguf_add_tensor(file.get(), t);
    };
    add("dense.weight", {64, 32}, true);
    add("dense.bias", {32}, false);
    add("embedding.weight", {64, 3}, true);
    add("rel_bias", {32, 3}, true);
    add("large.weight", {256, 64}, true);
    add("streaming.weight", {64, 17000}, true);
    add("norm.weight", {64}, false);
    add("norm.bias", {64}, false);
    add("conv.weight", {4, 4, 1, 8}, false);
    add("unaligned.weight", {17, 4}, false);
    require(gguf_write_to_file(file.get(), path.c_str(), false), "fixture write failed");
}
static void verify(const std::string &source, const std::string &output) {
    ModelLoader original, quantized;
    original.load(source); quantized.load(output);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> original_file(gguf_init_from_file(source.c_str(), {true, nullptr}), gguf_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> quantized_file(gguf_init_from_file(output.c_str(), {true, nullptr}), gguf_free);
    require(quantized.metadata("test.metadata") == "preserved", "metadata changed");
    const auto alignment_key = gguf_find_key(quantized_file.get(), "general.alignment");
    require(alignment_key >= 0 && gguf_get_val_u32(quantized_file.get(), alignment_key) == 32 &&
            gguf_get_alignment(quantized_file.get()) == 32, "output alignment was not normalized");
    const auto key = gguf_find_key(quantized_file.get(), "test.array");
    require(key >= 0 && gguf_get_arr_n(quantized_file.get(), key) == 3, "array metadata changed");
    const auto *array = static_cast<const int32_t *>(gguf_get_arr_data(quantized_file.get(), key));
    require(array[0] == 7 && array[1] == -3 && array[2] == 42, "array values changed");
    for (int64_t i = 0; i < gguf_get_n_tensors(original_file.get()); ++i) {
        const char *name = gguf_get_tensor_name(original_file.get(), i);
        auto *a = original.weight(name), *b = quantized.weight(name);
        require(std::memcmp(a->ne, b->ne, sizeof(a->ne)) == 0, "shape changed");
        const bool eligible = ggml_n_dims(a) > 1 && a->ne[0] % 32 == 0;
        require(b->type == (eligible ? GGML_TYPE_Q8_0 : a->type), "mixed precision policy violated");
        if (!eligible) {
            require(ggml_nbytes(a) == ggml_nbytes(b) && std::memcmp(a->data, b->data, ggml_nbytes(a)) == 0, "fallback tensor changed");
            continue;
        }
        std::vector<float> floats(ggml_nelements(a));
        for (size_t j = 0; j < floats.size(); ++j) floats[j] = scalar(a, j);
        std::vector<uint8_t> expected(ggml_nbytes(b));
        // Independent canonical reference entry point, not the production chunk wrapper.
        ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(floats.data(), expected.data(), floats.size());
        require(std::memcmp(expected.data(), b->data, expected.size()) == 0, "noncanonical Q8_0 bytes");
        std::vector<float> decoded(floats.size());
        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(b->data, decoded.data(), decoded.size());
        for (size_t j = 0; j < decoded.size(); ++j)
            require(scalar(b, j) == decoded[j], "Q8 scalar/embedding decode mismatch");
    }
    TensorOps ops(quantized, 2);
    Matrix input(3, 64);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = std::sin(static_cast<float>(i));
    auto actual = ops.linear(input, "dense");
    for (int r = 0; r < input.rows; ++r) for (int c = 0; c < actual.cols; ++c) {
        float expected = scalar(quantized.weight("dense.bias"), c);
        for (int k = 0; k < input.cols; ++k) expected += input[r][k] * scalar(quantized.weight("dense.weight"), c * input.cols + k);
        require(std::abs(actual[r][c] - expected) < 0.005f, "native Q8 matmul mismatch");
    }
    // >2 MiB activation packing scratch: graph work must not use fixed arena spare.
    Matrix large(10000, 256);
    for (auto &v : large.data) v = 0.25f;
    const auto result = ops.linear(large, "large", false);
    require(result.rows == large.rows && result.cols == 64, "large matmul shape");
    for (int c = 0; c < result.cols; ++c) {
        float expected = 0;
        for (int k = 0; k < large.cols; ++k) expected += 0.25f * scalar(quantized.weight("large.weight"), c * large.cols + k);
        require(std::abs(result[0][c] - expected) < 0.002f && result[0][c] == result[9999][c], "large matmul mismatch");
    }
    std::cout << "canonical Q8 bytes, mixed precision, scalar reads, native matmul and large scratch passed\n";
}
int main(int argc, char **argv) {
    try {
        require(argc == 4, "usage: q8-test create|verify SOURCE OUTPUT");
        if (std::string(argv[1]) == "create") {
            create(argv[2], GGML_TYPE_F32);
            create(std::string(argv[2]) + ".align64.gguf", GGML_TYPE_F32, 64);
            create(std::string(argv[2]) + ".f16.gguf", GGML_TYPE_F16);
            create(argv[3], GGML_TYPE_Q8_0);
        } else verify(argv[2], argv[3]);
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
