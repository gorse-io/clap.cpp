#include "model-loader.h"
#include "graph-test-helpers.h"
#include "quantize.h"
#include <ggml.h>
#include <gguf.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clap::detail;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using File = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
static void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
static const char *dense_name(const std::string &arch) {
    if (arch == "mert")
        return "layers.0.attention.q_proj.weight";
    if (arch == "mert2")
        return "layers.0.attn.query_proj.weight";
    if (arch == "muq" || arch == "muq_mulan")
        return "conformer.layers.0.self_attn.linear_q.weight";
    return "text.encoder.layer.0.attention.self.query.weight";
}
static std::string narrow_name(const std::string &arch) {
    std::string name(dense_name(arch));
    name.replace(name.find(".0."), 3, ".1.");
    return name;
}
static void fixture(const std::string &path, const std::string &arch, ggml_type type, bool bundled = false) {
    Context ctx(ggml_init({32 * 1024 * 1024, nullptr, false}), ggml_free);
    File file(gguf_init_empty(), gguf_free);
    gguf_set_val_str(file.get(), "general.architecture", arch.c_str());
    gguf_set_val_str(file.get(), "test.metadata", "preserved");
    if (bundled)
        gguf_set_val_str(file.get(), "clamp3.audio_frontend", "mert");
    auto add = [&](const char *name, std::vector<int64_t> shape, ggml_type storage) {
        auto *t = ggml_new_tensor(ctx.get(), storage, int(shape.size()), shape.data());
        ggml_set_name(t, name);
        std::vector<float> data(ggml_nelements(t));
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = 0.13f * std::sin(float(i) * 0.17f) + 0.01f;
        if (storage == GGML_TYPE_F32)
            std::memcpy(t->data, data.data(), ggml_nbytes(t));
        else
            ggml_get_type_traits(storage)->from_float_ref(data.data(), t->data, data.size());
        gguf_add_tensor(file.get(), t);
    };
    add(dense_name(arch), {256, 5000}, type); // Crosses the bounded chunk boundary.
    add("norm.weight", {256}, GGML_TYPE_F32);
    add("frontend.filters", {256, 2}, GGML_TYPE_F32);
    add("feature_extractor.mel_scale.fb", {256, 2}, GGML_TYPE_F32);
    add("conv.kernel.weight", {3, 2, 4}, type);
    add(narrow_name(arch).c_str(), {17, 3}, type);
    if (bundled)
        add("mert.layers.0.attention.q_proj.weight", {256, 3}, type);
    require(gguf_write_to_file(file.get(), path.c_str(), false), "fixture write");
}
static void verify(const std::string &source, const std::string &output, const std::string &arch) {
    ModelLoader input, result;
    input.load(source, false);
    result.load(output, false);
    File file(gguf_init_from_file(output.c_str(), {true, nullptr}), gguf_free);
    require(gguf_get_val_u32(file.get(), gguf_find_key(file.get(), "general.file_type")) == 32,
            "BF16 model file type must be 32, not tensor type 30");
    require(result.metadata("test.metadata") == "preserved", "metadata lost");
    for (int64_t i = 0; i < gguf_get_n_tensors(file.get()); ++i) {
        const auto *name = gguf_get_tensor_name(file.get(), i);
        auto *a = input.weight(name), *b = result.weight(name);
        const bool eligible = std::string(name) == dense_name(arch) || std::string(name) == narrow_name(arch);
        require(b->type == (eligible ? GGML_TYPE_BF16 : a->type), std::string("BF16 storage policy: ") + name);
        if (!eligible) {
            require(ggml_nbytes(a) == ggml_nbytes(b) && std::memcmp(a->data, b->data, ggml_nbytes(a)) == 0,
                    std::string("preserved tensor changed: ") + name);
            continue;
        }
        std::vector<float> values(ggml_nelements(a)), decoded(values.size()), row(b->ne[0]);
        if (a->type == GGML_TYPE_F32)
            std::memcpy(values.data(), a->data, ggml_nbytes(a));
        else
            ggml_get_type_traits(a->type)->to_float(a->data, values.data(), values.size());
        std::vector<ggml_bf16_t> expected(values.size());
        ggml_fp32_to_bf16_row_ref(values.data(), expected.data(), values.size());
        require(std::memcmp(expected.data(), b->data, ggml_nbytes(b)) == 0, "noncanonical BF16 bytes");
        ggml_bf16_to_fp32_row(expected.data(), decoded.data(), decoded.size());
        for (size_t j = 0; j < values.size(); ++j)
            require(scalar(b, j) == decoded[j], "BF16 scalar decode");
        for (int64_t r : {int64_t(0), ggml_nrows(b) - 1}) {
            tensor_row(b, r, row.data());
            require(std::memcmp(row.data(), decoded.data() + r * b->ne[0], row.size() * sizeof(float)) == 0,
                    "BF16 row decode");
        }
    }
    ModelLoader inference;
    inference.load(output);
    GraphRuntime ops(inference, 2);
    HostMatrix x(3, 256);
    for (size_t i = 0; i < x.data.size(); ++i)
        x.data[i] = std::sin(float(i) * 0.071f) * 0.17f;
    auto y = graph_linear(ops, x, std::string(dense_name(arch)).substr(0, std::strlen(dense_name(arch)) - 7), false);
    auto *w = result.weight(dense_name(arch));
    require(inference.weight(dense_name(arch))->type == GGML_TYPE_BF16, "BF16 mapping expanded by inference");
    Graph graph(ops);
    auto rows = graph.run(graph.embedding(dense_name(arch), std::vector<int>{0, 2}));
    for (int r = 0; r < 2; ++r)
        for (int j = 0; j < 256; ++j)
            require(rows[r * 256 + j] == scalar(w, (r == 0 ? 0 : 2) * 256 + j), "BF16 graph embedding");
    for (int r = 0; r < x.rows; ++r)
        for (int c = 0; c < y.cols; ++c) {
            float expected = 0;
            for (int j = 0; j < x.cols; ++j)
                expected += x[r][j] * scalar(w, c * x.cols + j);
            require(std::abs(y[r][c] - expected) < 2e-5f, arch + " BF16 F32-activation matmul drift");
        }
}
int main(int argc, char **argv) {
    try {
        require(argc == 2 || argc == 3, "usage: bf16-test DIRECTORY [scalar|loader]");
        std::filesystem::create_directories(argv[1]);
        if (argc == 3) {
            const auto path = std::string(argv[1]) + "/direct.gguf";
            fixture(path, "clamp3", GGML_TYPE_BF16, true);
            if (std::string(argv[2]) == "loader") {
                ModelLoader loader;
                loader.load(path);
            } else {
                Context ctx(ggml_init({1024 * 1024, nullptr, false}), ggml_free);
                auto *t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, 17, 2);
                std::vector<float> values(34, 0.123456f), row(17);
                ggml_fp32_to_bf16_row_ref(values.data(), static_cast<ggml_bf16_t *>(t->data), values.size());
                require(scalar(t, 31) == ggml_bf16_to_fp32(static_cast<ggml_bf16_t *>(t->data)[31]), "BF16 scalar");
                tensor_row(t, 1, row.data());
                require(row[14] == scalar(t, 31), "BF16 row");
            }
            return 0;
        }
        for (const std::string arch : {"clap", "clamp3", "mert", "mert2", "muq", "muq_mulan"})
            for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
                const auto source = std::string(argv[1]) + "/" + arch + std::to_string(type) + ".gguf";
                fixture(source, arch, type, arch == "clamp3");
                const auto output = source + ".bf16";
                std::filesystem::remove(output);
                require(cmd_quantize(source, output, "bf16") == 0, arch + " BF16 conversion refused");
                verify(source, output, arch);
                for (const std::string preset : {"Q8_0", "Q4_K_M"}) {
                    const auto q = output + preset;
                    std::filesystem::remove(q);
                    require(cmd_quantize(output, q, preset) == 0, "BF16 floating source rejected: " + preset);
                    ModelLoader floating, packed;
                    floating.load(output, false);
                    packed.load(q, false);
                    auto *a = floating.weight(dense_name(arch)), *b = packed.weight(dense_name(arch));
                    require(b->type == (preset == "Q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_K), "BF16 source target type");
                    std::vector<float> values(ggml_nelements(a));
                    ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t *>(a->data), values.data(), values.size());
                    std::vector<uint8_t> expected(ggml_nbytes(b));
                    ggml_get_type_traits(b->type)->from_float_ref(values.data(), expected.data(), values.size());
                    require(std::memcmp(expected.data(), b->data, expected.size()) == 0,
                            "BF16 source noncanonical Q bytes");
                }
                std::cout << arch << " " << type << " BF16 storage/scalar/row/graph and Q8/Q4 source passed\n";
            }
        // A pre-existing bundled BF16 MERT must load without block-quantization restrictions.
        const auto bundle = std::string(argv[1]) + "/bundle.gguf";
        fixture(bundle, "clamp3", GGML_TYPE_BF16, true);
        ModelLoader loader;
        loader.load(bundle);
        auto *weight = loader.weight("mert.layers.0.attention.q_proj.weight");
        require(std::isfinite(scalar(weight, 3)), "bundled BF16 MERT");
        GraphRuntime runtime(loader, 2);
        HostMatrix input(2, 256);
        for (auto &value : input.data)
            value = 0.25f;
        const auto output = graph_linear(runtime, input, "mert.layers.0.attention.q_proj", false);
        for (int c = 0; c < output.cols; ++c) {
            float expected = 0;
            for (int j = 0; j < input.cols; ++j)
                expected += 0.25f * scalar(weight, c * input.cols + j);
            require(std::abs(output[0][c] - expected) < 2e-5f && output[0][c] == output[1][c],
                    "bundled BF16 MERT graph");
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
