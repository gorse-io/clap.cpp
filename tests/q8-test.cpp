#include "model-loader.h"
#include "model-ops.h"
#include <ggml.h>
#include <gguf.h>
#include <cmath>
#include <cstdint>
#include <string>
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
static void create(const std::string &path, ggml_type type, uint32_t alignment = 32, const char *architecture = "clap", bool narrow = false) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({8 * 1024 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    gguf_set_val_str(file.get(), "general.architecture", architecture);
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
        if (narrow && shape[0] >= 256) shape[0] = 64;
        const auto storage_type = quantizable ? type :
            (shape.size() > 1 && type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
        auto *t = ggml_new_tensor(ctx.get(), storage_type, static_cast<int>(shape.size()), shape.data());
        ggml_set_name(t, name);
        std::vector<float> data(ggml_nelements(t));
        const size_t zero_block = quantizable ? (t->ne[0] % 256 == 0 ? 256 : 32) : 0;
        for (size_t i = zero_block; i < data.size(); ++i) data[i] = std::sin(static_cast<float>(i) * 0.13f) * 0.1f;
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
    add("text.encoder.layer.0.attention.self.value.weight", {256, 4}, true);
    add("text.encoder.layer.0.output.dense.weight", {512, 4}, true);
    add("text.encoder.layer.0.attention.output.dense.weight", {256, 4}, true);
    add("text_proj.weight", {256, 4}, true);
    add("text_proj.linear1.weight", {256, 4}, true);
    add("audio_proj.linear2.weight", {256, 4}, true);
    add("embedding.k.weight", {256, 3}, true);
    add("streaming.weight", {256, 5000}, true);
    add("norm.weight", {64}, false);
    add("norm.k.weight", {256}, false);
    add("norm.bias", {64}, false);
    add("conv.weight", {4, 4, 1, 8}, false);
    add("unaligned.weight", {17, 4}, false);
    require(gguf_write_to_file(file.get(), path.c_str(), false), "fixture write failed");
}
static void verify(const std::string &source, const std::string &output, bool k = false) {
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
    const auto ftype = gguf_find_key(quantized_file.get(), "general.file_type");
    require(ftype >= 0 && gguf_get_val_u32(quantized_file.get(), ftype) ==
            (k ? 15u : static_cast<uint32_t>(GGML_FTYPE_MOSTLY_Q8_0)), "wrong model preset metadata");
    if (k) require(quantized.metadata("clap.quantization_policy") ==
                   "q4_k_m-v1-all-v-ffn-down-projection-q6_k", "missing K policy metadata");
    int q4 = 0, q6 = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(original_file.get()); ++i) {
        const char *name = gguf_get_tensor_name(original_file.get(), i);
        auto *a = original.weight(name), *b = quantized.weight(name);
        require(std::memcmp(a->ne, b->ne, sizeof(a->ne)) == 0, "shape changed");
        const bool eligible = ggml_n_dims(a) > 1 && a->ne[0] % (k ? 256 : 32) == 0;
        const std::string n(name);
        const bool high = n.find(".attention.self.value.weight") != std::string::npos ||
            n == "text.encoder.layer.0.output.dense.weight" ||
            (original.metadata("general.architecture") == "clap" ?
                (n == "text_proj.linear1.weight" || n == "audio_proj.linear2.weight") : n == "text_proj.weight");
        const auto target = k ? (high ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K) : GGML_TYPE_Q8_0;
        require(b->type == (eligible ? target : a->type), "mixed precision policy violated");
        q4 += b->type == GGML_TYPE_Q4_K; q6 += b->type == GGML_TYPE_Q6_K;
        if (!eligible) {
            require(ggml_nbytes(a) == ggml_nbytes(b) && std::memcmp(a->data, b->data, ggml_nbytes(a)) == 0, "fallback tensor changed");
            continue;
        }
        std::vector<float> floats(ggml_nelements(a));
        for (size_t j = 0; j < floats.size(); ++j) floats[j] = scalar(a, j);
        std::vector<uint8_t> expected(ggml_nbytes(b));
        // Independent canonical reference entry point, not the production chunk wrapper.
        ggml_get_type_traits(b->type)->from_float_ref(floats.data(), expected.data(), floats.size());
        require(std::memcmp(expected.data(), b->data, expected.size()) == 0, "noncanonical quantized bytes");
        std::vector<float> decoded(floats.size());
        ggml_get_type_traits(b->type)->to_float(b->data, decoded.data(), decoded.size());
        for (size_t j = 0; j < decoded.size(); ++j)
            require(scalar(b, j) == decoded[j], "quantized scalar decode mismatch");
        std::vector<float> row(b->ne[0]);
        for (int64_t r = 0; r < ggml_nrows(b); ++r) {
            tensor_row(b, r, row.data());
            require(std::memcmp(row.data(), decoded.data() + r * b->ne[0], row.size() * sizeof(float)) == 0,
                    "quantized embedding row decode mismatch");
        }
    }
    require(!k || (q4 > 0 && q6 > 0), "K fixture must exercise both Q4_K and Q6_K");
    TensorOps ops(quantized, 2);
    Matrix input(3, 64);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = std::sin(static_cast<float>(i));
    auto actual = ops.linear(input, "dense");
    for (int r = 0; r < input.rows; ++r) for (int c = 0; c < actual.cols; ++c) {
        float expected = scalar(quantized.weight("dense.bias"), c);
        for (int k = 0; k < input.cols; ++k) expected += input[r][k] * scalar(quantized.weight("dense.weight"), c * input.cols + k);
        require(std::abs(actual[r][c] - expected) < 0.005f, "native Q8 matmul mismatch");
    }
    if (k) for (const auto *name : {"text.encoder.layer.0.attention.self.value",
                                    "text.encoder.layer.0.output.dense"}) {
        auto *weight = quantized.weight(std::string(name) + ".weight");
        Matrix x(3, static_cast<int>(weight->ne[0]));
        for (auto &v : x.data) v = 0.25f;
        const auto y = ops.linear(x, name, false);
        for (int c = 0; c < y.cols; ++c) {
            float expected = 0;
            for (int j = 0; j < x.cols; ++j) expected += 0.25f * scalar(weight, c * x.cols + j);
            require(std::abs(y[0][c] - expected) < 2e-5f && y[0][c] == y[2][c], "native Q6_K matmul mismatch");
        }
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
    std::cout << "canonical quantized bytes, mixed precision, scalar/row reads, native matmul and large scratch passed\n";
}
// Test-only canonical storage oracle exporter. Not linked into production CLI.
static void decode(const std::string &source, const std::string &destination) {
    ModelLoader loader; loader.load(source, false);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> input(gguf_init_from_file(source.c_str(), {true, nullptr}), gguf_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
    gguf_set_kv(output.get(), input.get());
    gguf_set_val_u32(output.get(), "general.alignment", gguf_get_alignment(output.get()));
    gguf_set_val_u32(output.get(), "general.file_type", GGML_FTYPE_ALL_F32);
    const auto count = gguf_get_n_tensors(input.get());
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({static_cast<size_t>(count + 1) * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    std::vector<std::vector<float>> values(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        const auto *w = loader.weight(gguf_get_tensor_name(input.get(), i));
        auto *t = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, GGML_MAX_DIMS, w->ne);
        ggml_set_name(t, w->name); values[i].resize(ggml_nelements(w)); t->data = values[i].data();
        // Use type traits directly, independently of the production row/scalar wrapper.
        if (w->type == GGML_TYPE_F32) std::memcpy(t->data, w->data, ggml_nbytes(w));
        else ggml_get_type_traits(w->type)->to_float(w->data, values[i].data(), ggml_nelements(w));
        gguf_add_tensor(output.get(), t);
    }
    require(gguf_write_to_file(output.get(), destination.c_str(), false), "oracle export failed");
}
int main(int argc, char **argv) {
    try {
        require(argc == 4, "usage: q8-test create|verify SOURCE OUTPUT");
        if (std::string(argv[1]) == "create") {
            create(argv[2], GGML_TYPE_F32);
            create(std::string(argv[2]) + ".align64.gguf", GGML_TYPE_F32, 64);
            create(std::string(argv[2]) + ".f16.gguf", GGML_TYPE_F16);
            create(argv[3], GGML_TYPE_Q8_0);
            create(std::string(argv[2]) + ".clamp3.gguf", GGML_TYPE_F32, 32, "clamp3");
            create(std::string(argv[2]) + ".narrow.gguf", GGML_TYPE_F32, 32, "clap", true);
        } else if (std::string(argv[1]) == "decode") decode(argv[2], argv[3]);
        else verify(argv[2], argv[3], std::string(argv[1]) == "verify-k");
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
