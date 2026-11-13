#include "model-loader.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace clap::detail {
ModelLoader::~ModelLoader() {
    if (weights)
        ggml_free(weights);
    if (gguf)
        gguf_free(gguf);
#ifndef _WIN32
    if (mapping && mapping != MAP_FAILED)
        munmap(mapping, mapping_size);
#endif
}

ggml_tensor *ModelLoader::weight(const std::string &name) const {
    if (source)
        return source->weight(tensor_prefix + name);
    const auto found = tensor_index.find(name);
    auto *w = found == tensor_index.end() ? nullptr : found->second;
    if (w == nullptr)
        throw std::runtime_error("CLAP: missing tensor " + name);
    return w;
}
ggml_tensor *ModelLoader::create_tensor(const std::string &name, std::initializer_list<int64_t> dims,
                                        TensorRequirement requirement) const {
    if (dims.size() == 0 || dims.size() > GGML_MAX_DIMS)
        throw std::runtime_error("CLAP: invalid tensor contract: " + name);
    for (const auto dim : dims)
        if (dim <= 0)
            throw std::runtime_error("CLAP: invalid tensor contract: " + name);
    if (source)
        return source->create_tensor(tensor_prefix + name, dims, requirement);
    if (requirement == TensorRequirement::Optional && tensor_index.find(name) == tensor_index.end())
        return nullptr;
    auto *w = weight(name);
    if (policy.validate_tensor_contract)
        policy.validate_tensor_contract(w, dims);
    int i = 0;
    for (auto d : dims) {
        if (w->ne[i++] != d)
            throw std::runtime_error("CLAP: wrong tensor shape: " + name);
    }
    while (i < GGML_MAX_DIMS)
        if (w->ne[i++] != 1)
            throw std::runtime_error("CLAP: wrong tensor rank: " + name);
    validated_shapes.insert(name);
    return w;
}

void ModelLoader::validate_tensor_coverage() const {
    if (!policy.unrecognized_tensor_error)
        return;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const std::string name = gguf_get_tensor_name(gguf, i);
        if (validated_shapes.count(name) == 0)
            throw std::runtime_error(std::string(policy.unrecognized_tensor_error) + name);
    }
}

std::string ModelLoader::metadata(const char *key) const {
    if (source)
        return source->metadata((metadata_prefix + key).c_str());
    const auto id = gguf_find_key(gguf, key);
    if (!(id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_STRING))
        throw std::runtime_error(std::string("CLAP: missing string metadata: ") + key);
    return std::string(gguf_get_val_str(gguf, id));
}

bool ModelLoader::has_metadata_prefix(const std::string &prefix) const {
    if (source)
        return source->has_metadata_prefix(metadata_prefix + prefix);
    for (int64_t i = 0; i < gguf_get_n_kv(gguf); ++i)
        if (std::string(gguf_get_key(gguf, i)).rfind(prefix, 0) == 0)
            return true;
    return false;
}
bool ModelLoader::has_tensor_prefix(const std::string &prefix) const {
    if (source)
        return source->has_tensor_prefix(tensor_prefix + prefix);
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i)
        if (std::string(gguf_get_tensor_name(gguf, i)).rfind(prefix, 0) == 0)
            return true;
    return false;
}

void ModelLoader::load(const std::string &path, bool inference, TensorLoadPolicySelector select_policy) {
    gguf = gguf_init_from_file(path.c_str(), {true, &weights});
    if (!(gguf && weights))
        throw std::runtime_error("CLAP: cannot load GGUF " + path);
    for (auto *tensor = ggml_get_first_tensor(weights); tensor; tensor = ggml_get_next_tensor(weights, tensor))
        tensor_index.emplace(ggml_get_name(tensor), tensor);
#ifndef _WIN32
    const int fd = open(path.c_str(), O_RDONLY);
    if (!(fd >= 0))
        throw std::runtime_error("CLAP: cannot open weights");
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        throw std::runtime_error("CLAP: cannot stat weights");
    }
    mapping_size = static_cast<size_t>(st.st_size);
    mapping = mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED)
        throw std::runtime_error("CLAP: cannot mmap weights");
#else
    std::ifstream stream(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!(static_cast<bool>(stream)))
        throw std::runtime_error("CLAP: cannot open weights");
    mapping_size = static_cast<size_t>(stream.tellg());
    storage.resize(mapping_size);
    stream.seekg(0);
    stream.read(storage.data(), static_cast<std::streamsize>(storage.size()));
    if (!(static_cast<bool>(stream)))
        throw std::runtime_error("CLAP: cannot read weights");
    mapping = storage.data();
#endif
    policy = select_policy ? select_policy(*this) : TensorLoadPolicy{};
    const auto data_offset = gguf_get_data_offset(gguf);
    if (!(data_offset <= mapping_size))
        throw std::runtime_error("CLAP: truncated GGUF data");
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        auto *w = weight(gguf_get_tensor_name(gguf, i));
        if (policy.validate_tensor_type)
            policy.validate_tensor_type(w);
        if (!(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16 ||
              w->type == GGML_TYPE_Q8_0 || w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q6_K))
            throw std::runtime_error("CLAP: only F32/F16/BF16/Q8_0/Q4_K/Q6_K weights supported");
        const size_t offset = gguf_get_tensor_offset(gguf, i);
        if (!(offset <= mapping_size - data_offset && ggml_nbytes(w) <= mapping_size - data_offset - offset))
            throw std::runtime_error("CLAP: truncated tensor data");
        w->data = static_cast<char *>(mapping) + data_offset + offset;
        if (policy.validate_tensor_data)
            policy.validate_tensor_data(w);
        if (inference && w->type == GGML_TYPE_F16 && policy.promote_f16 && policy.promote_f16(w)) {
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
