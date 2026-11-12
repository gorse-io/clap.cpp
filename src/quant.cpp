#include "model-loader.h"
#include "quant.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <vector>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#else
#include <unistd.h>
#endif

namespace clap::detail {
float scalar(const ggml_tensor *t, size_t i) {
    if (t->type == GGML_TYPE_F32)
        return static_cast<const float *>(t->data)[i];
    if (t->type == GGML_TYPE_F16)
        return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(t->data)[i]);
    if (t->type == GGML_TYPE_BF16)
        return ggml_bf16_to_fp32(static_cast<const ggml_bf16_t *>(t->data)[i]);
    if (!(t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q6_K))
        throw std::runtime_error("CLAP: unsupported scalar tensor type");
    // Decode only the containing block (embeddings, relative biases, or norms),
    // never expand an entire mapped weight tensor/model.
    float block[256];
    const size_t width = static_cast<size_t>(ggml_blck_size(t->type));
    const auto *data = static_cast<const uint8_t *>(t->data) + (i / width) * ggml_type_size(t->type);
    ggml_get_type_traits(t->type)->to_float(data, block, width);
    return block[i % width];
}
void tensor_row(const ggml_tensor *t, int64_t row, float *output) {
    if (!(row >= 0 && row < ggml_nrows(t)))
        throw std::runtime_error("CLAP: embedding row out of range");
    const auto *data = static_cast<const uint8_t *>(t->data) + row * t->nb[1];
    if (t->type == GGML_TYPE_F32)
        std::copy_n(reinterpret_cast<const float *>(data), t->ne[0], output);
    else if (t->type == GGML_TYPE_F16)
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(data), output, t->ne[0]);
    else {
        const auto *traits = ggml_get_type_traits(t->type);
        if (!(traits && traits->to_float))
            throw std::runtime_error("CLAP: unsupported embedding tensor type");
        traits->to_float(data, output, t->ne[0]);
    }
}

bool encoder_architecture(const std::string &architecture) {
    return architecture == "mert" || architecture == "mert2" || architecture == "muq" || architecture == "muq_mulan";
}
// Only native dense/embedding layouts are block quantizable. In particular,
// Conv1d [kernel,in,out], Conv2d, GRN and mel filters must remain floating point,
// even when their leading dimension happens to be divisible by a block size.
bool encoder_quantizable(const ggml_tensor *w, const std::string &architecture) {
    if (ggml_n_dims(w) != 2)
        return false;
    const std::string name(w->name);
    static const std::regex mert(
        R"((projection\.projection|layers\.[0-9]+\.(attention\.(q_proj|k_proj|v_proj|out_proj)|ffn\.(in|out)))\.weight)");
    static const std::regex mert2(
        R"((subsampling_module\.[0-9]+\.convnext_layers\.[0-9]+\.pointwise_block\.(1|4)|layers\.[0-9]+\.(attn\.(query_proj|key_proj|value_proj|out_proj)|ffn[12]\.w_[12]))\.weight)");
    static const std::regex muq(
        R"((conv\.linear|conformer\.layers\.[0-9]+\.(self_attn\.linear_(q|k|v|out)|ffn[12]\.(intermediate_dense|output_dense)))\.weight)");
    static const std::regex joint(
        R"((text\.embeddings\.(word_embeddings|position_embeddings|token_type_embeddings)|text\.encoder\.layer\.[0-9]+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)|adapter\.layers\.[0-9]+\.(0\.(to_q|to_kv|to_out\.0)|1\.(1|4))|audio_proj|audio_latent|text_proj)\.weight)");
    if (architecture == "mert")
        return std::regex_match(name, mert);
    if (architecture == "mert2")
        return std::regex_match(name, mert2);
    if (architecture == "muq" || architecture == "muq_mulan")
        return std::regex_match(name, muq) || (architecture == "muq_mulan" && std::regex_match(name, joint));
    return false;
}
static bool encoder_sensitive(const std::string &name, const std::string &architecture) {
    const auto ends = [&](const std::string &suffix) {
        return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (architecture == "mert")
        return name == "projection.projection.weight" || ends(".attention.v_proj.weight") || ends(".ffn.out.weight");
    if (architecture == "mert2")
        return ends(".attn.value_proj.weight") || ends(".ffn1.w_2.weight") || ends(".ffn2.w_2.weight") ||
               ends(".pointwise_block.4.weight");
    return name == "conv.linear.weight" || ends(".self_attn.linear_v.weight") || ends(".ffn1.output_dense.weight") ||
           ends(".ffn2.output_dense.weight") || ends(".attention.self.value.weight") ||
           (ends(".output.dense.weight") && !ends(".attention.output.dense.weight")) || ends(".0.to_kv.weight") ||
           ends(".1.4.weight") || name == "audio_proj.weight" || name == "audio_latent.weight" ||
           name == "text_proj.weight";
}

QuantizeOutput::QuantizeOutput(const std::filesystem::path &destination) : destination_(destination) {
    std::random_device random;
    for (int attempt = 0; attempt < 100; ++attempt) {
        temporary_ = destination;
        temporary_ += ".tmp-" + std::to_string(random()) + "-" + std::to_string(random());
        int fd = -1;
#ifdef _WIN32
        const int error = _wsopen_s(&fd, temporary_.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _SH_DENYRW,
                                    _S_IREAD | _S_IWRITE);
#else
        fd = ::open(temporary_.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
        const int error = fd < 0 ? errno : 0;
#endif
        if (fd < 0) {
            if (error == EEXIST)
                continue;
            throw std::system_error(error, std::generic_category(), "cannot create temporary output");
        }
#ifdef _WIN32
        stream_ = _fdopen(fd, "wb");
#else
        stream_ = ::fdopen(fd, "wb");
#endif
        if (!stream_) {
            const int error = errno;
#ifdef _WIN32
            _close(fd);
#else
            ::close(fd);
#endif
            std::error_code ignored;
            std::filesystem::remove(temporary_, ignored);
            throw std::system_error(error, std::generic_category(), "cannot open temporary stream");
        }
        return;
    }
    throw std::runtime_error("cannot create a unique temporary output");
}

QuantizeOutput::~QuantizeOutput() {
    if (stream_)
        std::fclose(stream_);
    std::error_code ignored;
    std::filesystem::remove(temporary_, ignored);
}

void QuantizeOutput::write(const void *data, size_t size) {
    if (!stream_ || std::fwrite(data, 1, size, stream_) != size)
        throw std::runtime_error("cannot write output");
}

void QuantizeOutput::close() {
    if (!stream_)
        throw std::runtime_error("output already closed");
    auto *stream = stream_;
    stream_ = nullptr;
    if (std::fclose(stream) != 0)
        throw std::runtime_error("cannot finish output");
}

void QuantizeOutput::publish() {
    if (stream_)
        throw std::runtime_error("output must be closed before publication");
    std::filesystem::create_hard_link(temporary_, destination_);
    std::filesystem::remove(temporary_);
    temporary_.clear();
}
} // namespace clap::detail

namespace {
using File = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
// Q4_K_M is a model preset, not a ggml tensor type. ggml's ftype enum
// has no M variant; use llama.cpp's GGUF model-file value (15), not ggml's
// unrelated MOSTLY_Q4_K value (12).
constexpr uint32_t q4_k_m_file_type = 15;
// GGUF model file types are not ggml tensor/standalone ftype enum values.
constexpr uint32_t bf16_file_type = 32;
enum class Preset { BF16, Q8_0, Q4_K_M };
bool bf16_eligible(const ggml_tensor *w, const std::string &architecture) {
    const std::string name(w->name);
    return ggml_n_dims(w) == 2 && !(architecture == "clamp3" && name.rfind("mert.", 0) == 0) &&
           (!clap::detail::encoder_architecture(architecture) || clap::detail::encoder_quantizable(w, architecture)) &&
           name.rfind("frontend.", 0) != 0 && name.rfind("feature_extractor.", 0) != 0;
}
ggml_type target_type(const ggml_tensor *w, Preset preset, const std::string &architecture) {
    // BF16 is floating storage, not block quantization: no row alignment
    // restriction. Keep vectors, convolution kernels and frontend buffers.
    if (preset == Preset::BF16)
        return bf16_eligible(w, architecture) ? GGML_TYPE_BF16 : w->type;
    if (architecture == "clamp3" && std::string(w->name).rfind("mert.", 0) == 0)
        return w->type;
    if (clap::detail::encoder_architecture(architecture) && !clap::detail::encoder_quantizable(w, architecture))
        return w->type;
    const bool k = preset == Preset::Q4_K_M;
    const auto base = k ? GGML_TYPE_Q4_K : GGML_TYPE_Q8_0;
    if (ggml_n_dims(w) <= 1 || w->ne[0] % ggml_blck_size(base) != 0)
        return w->type;
    if (!k)
        return base;
    const std::string name(w->name);
    if (clap::detail::encoder_architecture(architecture))
        return clap::detail::encoder_sensitive(name, architecture) ? GGML_TYPE_Q6_K : base;
    const auto ends = [&](const std::string &suffix) {
        return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    // Encoder adaptation of llama.cpp's sensitive V/FFN-down roles: protect
    // ALL such layers rather than its decoder-specific layer-index schedule.
    const bool value = ends(".attention.self.value.weight");
    const bool down = ends(".output.dense.weight") && !ends(".attention.output.dense.weight");
    const bool projection = architecture == "clap"
                                ? (name.rfind("text_proj.", 0) == 0 || name.rfind("audio_proj.", 0) == 0) &&
                                      (ends(".linear1.weight") || ends(".linear2.weight"))
                                : (name == "text_proj.weight" || name == "audio_proj.weight");
    return value || down || projection ? GGML_TYPE_Q6_K : base;
}
void quantize(const std::filesystem::path &source, const std::filesystem::path &destination, Preset preset) {
    const bool k = preset == Preset::Q4_K_M;
    const bool bf16 = preset == Preset::BF16;
    if (std::filesystem::exists(std::filesystem::symlink_status(destination)))
        throw std::runtime_error("CLAP: output already exists; choose a new path");
    clap::detail::ModelLoader loader;
    loader.load(source.string(), false);
    const auto architecture = loader.metadata("general.architecture");
    if (architecture == "clap2")
        throw std::runtime_error("CLAP: CLAP2 quantization presets are not supported; use converted F32/F16 weights");
    if (!(architecture == "clap" || architecture == "clamp3" || clap::detail::encoder_architecture(architecture)))
        throw std::runtime_error("CLAP: unsupported architecture");
    File input(gguf_init_from_file(source.string().c_str(), {true, nullptr}), gguf_free);
    File output(gguf_init_empty(), gguf_free);
    if (!(input && output))
        throw std::runtime_error("CLAP: cannot read GGUF metadata");
    gguf_set_kv(output.get(), input.get());
    // gguf_set_kv copies metadata, not the writer's internal alignment.
    // Normalize the advertised value to the actual tensor/metadata layout.
    gguf_set_val_u32(output.get(), "general.alignment", gguf_get_alignment(output.get()));
    gguf_set_val_u32(output.get(), "general.file_type",
                     bf16 ? bf16_file_type : (k ? q4_k_m_file_type : static_cast<uint32_t>(GGML_FTYPE_MOSTLY_Q8_0)));
    if (k)
        gguf_set_val_str(output.get(), "clap.quantization_policy",
                         clap::detail::encoder_architecture(architecture)
                             ? "q4_k_m-v2-encoder-dense-v-ffn-down-projection-q6_k"
                             : "q4_k_m-v1-all-v-ffn-down-projection-q6_k");
    gguf_set_val_u32(output.get(), "general.quantization_version", GGML_QNT_VERSION);
    size_t count = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(input.get()); ++i) {
        const auto *w = loader.weight(gguf_get_tensor_name(input.get(), i));
        if (!(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16))
            throw std::runtime_error("CLAP: quantization requires F32/F16/BF16 input; refusing requantization");
        gguf_add_tensor(output.get(), w);
        const auto target = target_type(w, preset, architecture);
        if (target != w->type)
            gguf_set_tensor_type(output.get(), w->name, target);
        if (target != w->type || (bf16 && bf16_eligible(w, architecture)))
            ++count;
    }
    if (!(count > 0))
        throw std::runtime_error(
            std::string("CLAP: ") +
            (bf16 ? "no BF16-compatible tensors" : (k ? "no Q4_K-compatible tensors" : "no Q8_0-compatible tensors")));
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
            const auto target = target_type(w, preset, architecture);
            if (target != w->type) {
                // Bound scratch by approximately one million floats (or one row).
                const int64_t rows_per_chunk = std::max<int64_t>(1, (1024 * 1024) / w->ne[0]);
                const int64_t chunk_elements = std::min(rows_per_chunk, ggml_nrows(w)) * w->ne[0];
                std::vector<float> values(chunk_elements);
                std::vector<uint8_t> packed(ggml_row_size(target, chunk_elements));
                for (int64_t row = 0; row < ggml_nrows(w); row += rows_per_chunk) {
                    const int64_t rows = std::min(rows_per_chunk, ggml_nrows(w) - row);
                    const int64_t elements = rows * w->ne[0];
                    const auto *data = static_cast<const uint8_t *>(w->data) + row * w->nb[1];
                    if (w->type == GGML_TYPE_F16)
                        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(data), values.data(), elements);
                    else if (w->type == GGML_TYPE_BF16)
                        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(data), values.data(), elements);
                    else
                        std::copy_n(reinterpret_cast<const float *>(data), elements, values.data());
                    const auto bytes =
                        ggml_quantize_chunk(target, values.data(), packed.data(), 0, rows, w->ne[0], nullptr);
                    if (bytes != ggml_row_size(target, elements))
                        throw std::runtime_error("CLAP: quantized byte count mismatch");
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
    std::cout << "Quantized " << count << " tensors to " << (bf16 ? "BF16" : (k ? "Q4_K_M" : "Q8_0"))
              << "; ineligible tensors kept unchanged\n";
}
} // namespace
int cmd_quantize(const std::string &input, const std::string &output, const std::string &type) {
    try {
        std::string preset(type);
        std::transform(preset.begin(), preset.end(), preset.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (!(preset == "BF16" || preset == "Q8_0" || preset == "Q4_K_M"))
            throw std::runtime_error("CLAP: unsupported quantization preset");
        quantize(input, output, preset == "BF16" ? Preset::BF16 : (preset == "Q4_K_M" ? Preset::Q4_K_M : Preset::Q8_0));
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
