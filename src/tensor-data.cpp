#include "tensor-data.h"

#include <algorithm>
#include <stdexcept>

namespace clap::detail {
HostMatrix::HostMatrix(int r, int c) : rows(r), cols(c), data() {
    if (!(r > 0 && c > 0))
        throw std::runtime_error("CLAP: invalid matrix shape");
    data.resize(static_cast<size_t>(r) * c);
}
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
} // namespace clap::detail
