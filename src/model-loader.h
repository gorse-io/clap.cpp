#pragma once

#include <ggml.h>
#include <gguf.h>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_set>
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
    bool f32_activation_compute() const { return mulan; }
    void validate_tensor_coverage() const;
private:
    bool mulan = false;
    mutable std::unordered_set<std::string> validated_shapes;
    gguf_context *gguf = nullptr;
    ggml_context *weights = nullptr;
    void *mapping = nullptr;
    size_t mapping_size = 0;
    // MuQ F16 is a storage format, not F16 activation compute. Owning promoted
    // buffers here preserves stable tensor identities and facade/backend lifetime.
    std::vector<std::vector<float>> promoted_weights;
#ifdef _WIN32
    std::vector<char> storage;
#endif
};
} // namespace clap::detail
