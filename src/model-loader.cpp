#include "model-loader.h"
#include <stdexcept>
#include <fstream>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace clap::detail {
ModelLoader::~ModelLoader() {
    if (weights) ggml_free(weights);
    if (gguf) gguf_free(gguf);
#ifndef _WIN32
    if (mapping && mapping != MAP_FAILED) munmap(mapping, mapping_size);
#endif
}

ggml_tensor *ModelLoader::weight(const std::string &name) const {
    auto *w = ggml_get_tensor(weights, name.c_str());
    if (w == nullptr)
        throw std::runtime_error("CLAP: missing tensor " + name);
    return w;
}
void ModelLoader::shape(const std::string &name, std::initializer_list<int64_t> dims) const {
    const auto *w = weight(name);
    int i = 0;
    for (auto d : dims) { if (w->ne[i++] != d) throw std::runtime_error("CLAP: wrong tensor shape: " + name); }
    while (i < GGML_MAX_DIMS) if (w->ne[i++] != 1) throw std::runtime_error("CLAP: wrong tensor rank: " + name);
}
void ModelLoader::linear_shape(const std::string &name, int in, int out, bool bias) const {
    shape(name + ".weight", {in, out});
    if (bias) shape(name + ".bias", {out});
}
void ModelLoader::norm_shape(const std::string &name, int dim) const {
    shape(name + ".weight", {dim}); shape(name + ".bias", {dim});
}

std::string ModelLoader::metadata(const char *key) const {
    const auto id = gguf_find_key(gguf, key);
    if (!(id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_STRING))
        throw std::runtime_error(std::string("CLAP: missing string metadata: ") + key);
    return std::string(gguf_get_val_str(gguf, id));
}

void ModelLoader::load(const std::string &path) {
    gguf = gguf_init_from_file(path.c_str(), {true, &weights});
    if (!(gguf && weights))
        throw std::runtime_error("CLAP: cannot load GGUF " + path);
#ifndef _WIN32
    const int fd = open(path.c_str(), O_RDONLY);
    if (!(fd >= 0))
        throw std::runtime_error("CLAP: cannot open weights");
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); throw std::runtime_error("CLAP: cannot stat weights"); }
    mapping_size = static_cast<size_t>(st.st_size);
    mapping = mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED)
        throw std::runtime_error("CLAP: cannot mmap weights");
#else
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!(static_cast<bool>(stream)))
        throw std::runtime_error("CLAP: cannot open weights");
    mapping_size = static_cast<size_t>(stream.tellg()); storage.resize(mapping_size);
    stream.seekg(0); stream.read(storage.data(), static_cast<std::streamsize>(storage.size()));
    if (!(static_cast<bool>(stream)))
        throw std::runtime_error("CLAP: cannot read weights");
    mapping = storage.data();
#endif
    const auto architecture = gguf_find_key(gguf, "general.architecture");
    const bool muq = architecture >= 0 && gguf_get_kv_type(gguf, architecture) == GGUF_TYPE_STRING &&
                     metadata("general.architecture") == "muq";
    const auto data_offset = gguf_get_data_offset(gguf);
    if (!(data_offset <= mapping_size))
        throw std::runtime_error("CLAP: truncated GGUF data");
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        auto *w = weight(gguf_get_tensor_name(gguf, i));
        // MuQ convolution layouts are validated only for floating-point storage.
        if (muq && w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16)
            throw std::runtime_error("CLAP: MuQ requires F32/F16 weights; quantized MuQ is unsupported");
        if (!(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_Q8_0 ||
              w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q6_K))
            throw std::runtime_error("CLAP: only F32/F16/Q8_0/Q4_K/Q6_K weights supported");
        const size_t offset = gguf_get_tensor_offset(gguf, i);
        if (!(offset <= mapping_size - data_offset && ggml_nbytes(w) <= mapping_size - data_offset - offset))
            throw std::runtime_error("CLAP: truncated tensor data");
        w->data = static_cast<char *>(mapping) + data_offset + offset;
        if (muq && w->type == GGML_TYPE_F16) {
            // GGML's F16 matmul packs F32 inputs to F16 too. MuQ's learned
            // feature norms amplify that extra rounding; preserve the
            // storage-rounded, F32-activation reference by promoting weights.
            promoted_weights.emplace_back(static_cast<size_t>(ggml_nelements(w)));
            auto &values = promoted_weights.back();
            ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(w->data), values.data(), ggml_nelements(w));
            w->type = GGML_TYPE_F32;
            w->nb[0] = sizeof(float);
            for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension)
                w->nb[dimension] = w->nb[dimension - 1] * w->ne[dimension - 1];
            w->data = values.data();
        }
    }
}
} // namespace clap::detail
