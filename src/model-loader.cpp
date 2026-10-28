#include "model-loader.h"
#include "quantization-policy.h"
#include "model-ops.h"
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <cstring>
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
    if (source) return source->weight(tensor_prefix + name);
    auto *w = ggml_get_tensor(weights, name.c_str());
    if (w == nullptr)
        throw std::runtime_error("CLAP: missing tensor " + name);
    return w;
}
void ModelLoader::shape(const std::string &name, std::initializer_list<int64_t> dims) const {
    if (source) { source->shape(tensor_prefix + name, dims); return; }
    const auto *w = weight(name);
    validated_shapes.insert(name);
    if (mulan && dims.size()==1 && w->type!=GGML_TYPE_F32)
        throw std::runtime_error("CLAP: MuLan vectors require F32: " + name);
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

void ModelLoader::validate_tensor_coverage(const std::string &prefix) const {
    if (!f32_activations && !bundled) return;
    for (int64_t i=0;i<gguf_get_n_tensors(gguf);++i) {
        const std::string name=gguf_get_tensor_name(gguf,i);
        if (!prefix.empty() && name.rfind(prefix, 0) != 0) continue;
        if (validated_shapes.count(name)==0) throw std::runtime_error(std::string(mulan ? "CLAP: unrecognized MuLan tensor: " : "CLAP: unrecognized model tensor: ") + name);
    }
}

std::string ModelLoader::metadata(const char *key) const {
    if (source) return source->metadata((metadata_prefix + key).c_str());
    const auto id = gguf_find_key(gguf, key);
    if (!(id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_STRING))
        throw std::runtime_error(std::string("CLAP: missing string metadata: ") + key);
    return std::string(gguf_get_val_str(gguf, id));
}

bool ModelLoader::has_metadata(const char *key) const {
    if (source) return source->has_metadata((metadata_prefix + key).c_str());
    return gguf_find_key(gguf, key) >= 0;
}

nlohmann::json ModelLoader::metadata_value(const char *key) const {
    if (source) return source->metadata_value((metadata_prefix + key).c_str());
    const auto id = gguf_find_key(gguf, key);
    if (id < 0) throw std::runtime_error(std::string("CLAP: missing metadata: ") + key);
    const auto type = gguf_get_kv_type(gguf, id);
    if (type == GGUF_TYPE_ARRAY) {
        const auto element_type = gguf_get_arr_type(gguf, id);
        const size_t count = gguf_get_arr_n(gguf, id);
        auto values = nlohmann::json::array();
        if (element_type == GGUF_TYPE_STRING) {
            for (size_t i = 0; i < count; ++i) values.push_back(gguf_get_arr_str(gguf, id, i));
            return values;
        }
        const auto *data = static_cast<const char *>(gguf_get_arr_data(gguf, id));
        const auto append = [&](auto sample) {
            using T = decltype(sample);
            for (size_t i = 0; i < count; ++i) {
                T value;
                std::memcpy(&value, data + i * sizeof(T), sizeof(T));
                values.push_back(value);
            }
        };
        switch (element_type) {
            case GGUF_TYPE_UINT8: append(uint8_t{}); break;
            case GGUF_TYPE_INT8: append(int8_t{}); break;
            case GGUF_TYPE_UINT16: append(uint16_t{}); break;
            case GGUF_TYPE_INT16: append(int16_t{}); break;
            case GGUF_TYPE_UINT32: append(uint32_t{}); break;
            case GGUF_TYPE_INT32: append(int32_t{}); break;
            case GGUF_TYPE_UINT64: append(uint64_t{}); break;
            case GGUF_TYPE_INT64: append(int64_t{}); break;
            case GGUF_TYPE_FLOAT32: append(float{}); break;
            case GGUF_TYPE_FLOAT64: append(double{}); break;
            case GGUF_TYPE_BOOL:
                for (size_t i = 0; i < count; ++i) values.push_back(data[i] != 0);
                break;
            default: throw std::runtime_error(std::string("CLAP: unsupported metadata array: ") + key);
        }
        return values;
    }
    switch (type) {
        case GGUF_TYPE_STRING: return gguf_get_val_str(gguf, id);
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(gguf, id);
        case GGUF_TYPE_INT8: return gguf_get_val_i8(gguf, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(gguf, id);
        case GGUF_TYPE_INT16: return gguf_get_val_i16(gguf, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(gguf, id);
        case GGUF_TYPE_INT64: return gguf_get_val_i64(gguf, id);
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(gguf, id);
        case GGUF_TYPE_BOOL: return gguf_get_val_bool(gguf, id);
        default: throw std::runtime_error(std::string("CLAP: unsupported metadata: ") + key);
    }
}

int64_t ModelLoader::integer(const char *key) const {
    const auto value = metadata_value(key);
    if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<uint64_t>() > INT64_MAX))
        throw std::runtime_error(std::string("CLAP: expected integer metadata: ") + key);
    return value.get<int64_t>();
}
double ModelLoader::number(const char *key) const {
    const auto value = metadata_value(key);
    if (!value.is_number() || !std::isfinite(value.get<double>()))
        throw std::runtime_error(std::string("CLAP: expected finite numeric metadata: ") + key);
    return value.get<double>();
}
bool ModelLoader::boolean(const char *key) const {
    const auto value = metadata_value(key);
    if (!value.is_boolean()) throw std::runtime_error(std::string("CLAP: expected boolean metadata: ") + key);
    return value.get<bool>();
}

bool ModelLoader::has_metadata_prefix(const std::string &prefix) const {
    if (source) return source->has_metadata_prefix(metadata_prefix + prefix);
    for (int64_t i = 0; i < gguf_get_n_kv(gguf); ++i)
        if (std::string(gguf_get_key(gguf, i)).rfind(prefix, 0) == 0) return true;
    return false;
}
bool ModelLoader::has_tensor_prefix(const std::string &prefix) const {
    if (source) return source->has_tensor_prefix(tensor_prefix + prefix);
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i)
        if (std::string(gguf_get_tensor_name(gguf, i)).rfind(prefix, 0) == 0) return true;
    return false;
}

void ModelLoader::load(const std::string &path, bool inference) {
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
    std::ifstream stream(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
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
    mulan = architecture >= 0 && gguf_get_kv_type(gguf, architecture) == GGUF_TYPE_STRING && metadata("general.architecture") == "muq_mulan";
    const bool named = architecture >= 0 && gguf_get_kv_type(gguf, architecture) == GGUF_TYPE_STRING;
    bundled = named && metadata("general.architecture") == "clamp3" && has_metadata_prefix("clamp3.audio_frontend");
    const std::string arch = named ? metadata("general.architecture") : "";
    // EmbeddingGemma2 requires F32 activations even with F16/BF16 storage;
    // narrow activation packing can overflow its projection/attention stack.
    f32_activations = encoder_architecture(arch) || arch == "gemma-embedding2" || arch == "clip";
    const auto data_offset = gguf_get_data_offset(gguf);
    if (!(data_offset <= mapping_size))
        throw std::runtime_error("CLAP: truncated GGUF data");
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        auto *w = weight(gguf_get_tensor_name(gguf, i));
        const std::string tensor_name = gguf_get_tensor_name(gguf, i);
        if ((bundled && tensor_name.rfind("mert.", 0) == 0) &&
            w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16)
            throw std::runtime_error("CLAP: MERT supports F32/F16 weights only");
        if (encoder_architecture(arch) && ggml_is_quantized(w->type) && !encoder_quantizable(w, arch))
            throw std::runtime_error("CLAP: quantized non-dense/embedding tensor is unsupported: " + tensor_name);
        const bool gemma_storage = (arch == "gemma-embedding2" || arch == "clip") &&
            (w->type == GGML_TYPE_Q5_K || w->type == GGML_TYPE_BF16);
        if (!(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_Q8_0 ||
              w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q6_K || gemma_storage))
            throw std::runtime_error("CLAP: unsupported tensor storage type: " + tensor_name);
        const size_t offset = gguf_get_tensor_offset(gguf, i);
        if (!(offset <= mapping_size - data_offset && ggml_nbytes(w) <= mapping_size - data_offset - offset))
            throw std::runtime_error("CLAP: truncated tensor data");
        w->data = static_cast<char *>(mapping) + data_offset + offset;
        const std::string name=gguf_get_tensor_name(gguf,i);
        const bool muq_backbone = mulan && (name.rfind("conv.",0)==0 || name.rfind("conformer.",0)==0);
        if (mulan) {
            // Non-module frontend buffers and all 1D normalization/statistic
            // tensors are F32. Dense/embedding and MuQ kernels may be F16.
            if ((name.rfind("frontend.",0)==0 || name.find(".running_")!=std::string::npos || (muq_backbone && ggml_n_dims(w)==1)) && w->type!=GGML_TYPE_F32)
                throw std::runtime_error("CLAP: MuLan frontend/buffers/vectors require F32: " + name);
            const bool batch_variance = name.size()>=12 && name.compare(name.size()-12,12,".running_var")==0;
            // Scan each packed block once (scalar() would decode it per element).
            float block[256];
            const int64_t width = ggml_is_quantized(w->type) ? ggml_blck_size(w->type) : 1;
            for (int64_t start=0;start<ggml_nelements(w);start+=width) {
                if (width > 1) ggml_get_type_traits(w->type)->to_float(
                    static_cast<const char *>(w->data)+(start/width)*ggml_type_size(w->type),block,width);
                else block[0]=scalar(w,static_cast<size_t>(start));
                for (int64_t j=0;j<width;++j) {
                    const float value=block[j];
                    if (!std::isfinite(value)) throw std::runtime_error("CLAP: non-finite MuLan tensor: " + name);
                    if (batch_variance && value<0)
                        throw std::runtime_error("CLAP: negative MuLan batch variance: " + name);
                }
            }
        }
        if (inference && (muq || muq_backbone) && w->type == GGML_TYPE_F16) {
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
