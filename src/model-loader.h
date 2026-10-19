#pragma once

#include <ggml.h>
#include <gguf.h>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace clap::detail {

// Architecture-independent GGUF ownership, mapping, metadata and tensor checks.
// Keep this alive longer than all model architectures and tensor operations.
class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    ModelLoader(const ModelLoader &) = delete;
    ModelLoader &operator=(const ModelLoader &) = delete;

    void load(const std::string &path);
    std::string metadata(const char *key) const;
    ggml_tensor *weight(const std::string &name) const;
    void shape(const std::string &name, std::initializer_list<int64_t> dims) const;
    void linear_shape(const std::string &name, int in, int out, bool bias = true) const;
    void norm_shape(const std::string &name, int dim) const;
private:
    gguf_context *gguf = nullptr;
    ggml_context *weights = nullptr;
    void *mapping = nullptr;
    size_t mapping_size = 0;
#ifdef _WIN32
    std::vector<char> storage;
#endif
};
} // namespace clap::detail
