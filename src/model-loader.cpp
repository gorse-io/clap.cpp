#include "model-loader.h"
#include "impl.h"
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
    check(w != nullptr, "missing tensor " + name);
    return w;
}
void ModelLoader::shape(const std::string &name, std::initializer_list<int64_t> dims) const {
    const auto *w = weight(name);
    int i = 0;
    for (auto d : dims) { check(w->ne[i++] == d, "wrong tensor shape: " + name); }
    while (i < GGML_MAX_DIMS) check(w->ne[i++] == 1, "wrong tensor rank: " + name);
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
    check(id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_STRING, std::string("missing string metadata: ") + key);
    return std::string(gguf_get_val_str(gguf, id));
}

void ModelLoader::load(const std::string &path) {
    gguf = gguf_init_from_file(path.c_str(), {true, &weights});
    check(gguf && weights, "cannot load GGUF " + path);
#ifndef _WIN32
    const int fd = open(path.c_str(), O_RDONLY);
    check(fd >= 0, "cannot open weights");
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); check(false, "cannot stat weights"); }
    mapping_size = static_cast<size_t>(st.st_size);
    mapping = mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    check(mapping != MAP_FAILED, "cannot mmap weights");
#else
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    check(static_cast<bool>(stream), "cannot open weights");
    mapping_size = static_cast<size_t>(stream.tellg()); storage.resize(mapping_size);
    stream.seekg(0); stream.read(storage.data(), static_cast<std::streamsize>(storage.size()));
    check(static_cast<bool>(stream), "cannot read weights"); mapping = storage.data();
#endif
    const auto data_offset = gguf_get_data_offset(gguf);
    check(data_offset <= mapping_size, "truncated GGUF data");
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        auto *w = weight(gguf_get_tensor_name(gguf, i));
        check(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_Q8_0,
              "only F32/F16/Q8_0 weights supported");
        const size_t offset = gguf_get_tensor_offset(gguf, i);
        check(offset <= mapping_size - data_offset && ggml_nbytes(w) <= mapping_size - data_offset - offset, "truncated tensor data");
        w->data = static_cast<char *>(mapping) + data_offset + offset;
    }
}
} // namespace clap::detail
