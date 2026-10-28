#include "model-loader.h"
#include "model-ops.h"
#include <ggml.h>
#include <gguf.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    try {
        for (const char *architecture : {"gemma-embedding2", "clip"}) {
            for (const auto type : {GGML_TYPE_Q5_K, GGML_TYPE_BF16}) {
                const std::string path = std::string(argv[1]) + std::to_string(type) + ".gguf";
                std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024 * 1024, nullptr, false}), ggml_free);
                std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
                gguf_set_val_str(file.get(), "general.architecture", architecture);
                auto *w = ggml_new_tensor_2d(ctx.get(), type, 256, 2);
                ggml_set_name(w, "dense.weight");
                std::vector<float> values(512);
                for (size_t i = 0; i < values.size(); ++i) values[i] = 0.1f * std::sin(float(i));
                if (type == GGML_TYPE_BF16) ggml_fp32_to_bf16_row_ref(values.data(), static_cast<ggml_bf16_t *>(w->data), values.size());
                else ggml_quantize_chunk(type, values.data(), w->data, 0, 2, 256, nullptr);
                gguf_add_tensor(file.get(), w);
                if (!gguf_write_to_file(file.get(), path.c_str(), false)) return 2;
                clap::detail::ModelLoader loader;
                loader.load(path);
                std::vector<float> decoded(512), row(256);
                ggml_get_type_traits(type)->to_float(w->data, decoded.data(), decoded.size());
                for (size_t i = 0; i < decoded.size(); ++i)
                    if (clap::detail::scalar(loader.weight("dense.weight"), i) != decoded[i]) return 3;
                clap::detail::tensor_row(loader.weight("dense.weight"), 1, row.data());
                for (size_t i = 0; i < row.size(); ++i) if (row[i] != decoded[256 + i]) return 4;
                clap::detail::Matrix x(2, 256);
                for (auto &v : x.data) v = 0.25f;
                const auto result = clap::detail::TensorOps(loader, 2).linear(x, "dense", false);
                for (int c = 0; c < 2; ++c) {
                    float expected = 0;
                    for (int i = 0; i < 256; ++i) expected += decoded[c * 256 + i] * 0.25f;
                    if (std::abs(result[0][c] - expected) > 0.002f) return 5;
                }
            }
        }
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 6; }
    return 0;
}
