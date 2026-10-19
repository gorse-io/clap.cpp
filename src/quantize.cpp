#include "model-loader.h"
#include "impl.h"
#include "quantize-output.h"
#include "quantize.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using clap::detail::check;
using File = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
bool eligible(const ggml_tensor *w) {
    return ggml_n_dims(w) > 1 && w->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0) == 0;
}
void quantize(const std::filesystem::path &source, const std::filesystem::path &destination) {
    check(!std::filesystem::exists(std::filesystem::symlink_status(destination)),
          "output already exists; choose a new path");
    clap::detail::ModelLoader loader;
    loader.load(source.string());
    const auto architecture = loader.metadata("general.architecture");
    check(architecture == "clap" || architecture == "clamp3", "unsupported architecture");
    File input(gguf_init_from_file(source.string().c_str(), {true, nullptr}), gguf_free);
    File output(gguf_init_empty(), gguf_free);
    check(input && output, "cannot read GGUF metadata");
    gguf_set_kv(output.get(), input.get());
    // gguf_set_kv copies metadata, not the writer's internal alignment.
    // Normalize the advertised value to the actual tensor/metadata layout.
    gguf_set_val_u32(output.get(), "general.alignment", gguf_get_alignment(output.get()));
    gguf_set_val_u32(output.get(), "general.file_type", GGML_FTYPE_MOSTLY_Q8_0);
    gguf_set_val_u32(output.get(), "general.quantization_version", GGML_QNT_VERSION);
    size_t count = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(input.get()); ++i) {
        const auto *w = loader.weight(gguf_get_tensor_name(input.get(), i));
        check(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16,
              "quantization requires F32/F16 input; refusing requantization");
        gguf_add_tensor(output.get(), w);
        if (eligible(w)) {
            gguf_set_tensor_type(output.get(), w->name, GGML_TYPE_Q8_0);
            ++count;
        }
    }
    check(count > 0, "no Q8_0-compatible tensors");
    clap::detail::QuantizeOutput stream(destination);
    {
        std::vector<char> metadata(gguf_get_meta_size(output.get()));
        gguf_get_meta_data(output.get(), metadata.data());
        stream.write(metadata.data(), static_cast<std::streamsize>(metadata.size()));
        const auto alignment = gguf_get_alignment(output.get());
        std::vector<char> padding(alignment, 0);
        for (int64_t i = 0; i < gguf_get_n_tensors(input.get()); ++i) {
            const auto *w = loader.weight(gguf_get_tensor_name(input.get(), i));
            size_t written = 0;
            if (eligible(w)) {
                // Bound scratch by approximately one million floats (or one row).
                const int64_t rows_per_chunk = std::max<int64_t>(1, (1024 * 1024) / w->ne[0]);
                const int64_t chunk_elements = std::min(rows_per_chunk, ggml_nrows(w)) * w->ne[0];
                std::vector<float> values(chunk_elements);
                std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_Q8_0, chunk_elements));
                for (int64_t row = 0; row < ggml_nrows(w); row += rows_per_chunk) {
                    const int64_t rows = std::min(rows_per_chunk, ggml_nrows(w) - row);
                    const int64_t elements = rows * w->ne[0];
                    const auto *data = static_cast<const uint8_t *>(w->data) + row * w->nb[1];
                    if (w->type == GGML_TYPE_F16)
                        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(data), values.data(), elements);
                    else std::copy_n(reinterpret_cast<const float *>(data), elements, values.data());
                    const auto bytes = ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), packed.data(), 0, rows, w->ne[0], nullptr);
                    check(bytes == ggml_row_size(GGML_TYPE_Q8_0, elements), "quantized byte count mismatch");
                    stream.write(reinterpret_cast<const char *>(packed.data()), static_cast<std::streamsize>(bytes));
                    written += bytes;
                }
            } else {
                written = ggml_nbytes(w);
                stream.write(static_cast<const char *>(w->data), static_cast<std::streamsize>(written));
            }
            const auto pad = (alignment - written % alignment) % alignment;
            stream.write(padding.data(), static_cast<std::streamsize>(pad));
        }
        stream.close();
        stream.publish();
    }
    std::cout << "Quantized " << count << " tensors to Q8_0; ineligible tensors kept unchanged\n";
}
} // namespace
int run_quantize(const std::string &input, const std::string &output) {
    try { quantize(input, output); return 0; }
    catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
