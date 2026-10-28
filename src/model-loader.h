#pragma once

#include <nlohmann/json.hpp>
#include <ggml.h>
#include <gguf.h>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

namespace clap::detail {

// Architecture-independent GGUF ownership, mapping, metadata and tensor checks.
// Keep this alive longer than all model architectures and tensor operations.
class ModelLoader {
public:
    ModelLoader() = default;
    // Non-owning namespace view: the source mapping must outlive this view.
    ModelLoader(const ModelLoader &source, std::string tensor_prefix, std::string metadata_prefix)
        : source(&source), tensor_prefix(std::move(tensor_prefix)), metadata_prefix(std::move(metadata_prefix)) {}
    bool has_metadata_prefix(const std::string &prefix) const;
    bool has_tensor_prefix(const std::string &prefix) const;
    ~ModelLoader();
    ModelLoader(const ModelLoader &) = delete;
    ModelLoader &operator=(const ModelLoader &) = delete;

    // Storage tools must not promote F16 bytes before conversion/copying.
    void load(const std::string &path, bool inference = true);
    std::string metadata(const char *key) const;
    bool has_metadata(const char *key) const;
    nlohmann::json metadata_value(const char *key) const;
    int64_t integer(const char *key) const;
    double number(const char *key) const;
    bool boolean(const char *key) const;
    ggml_tensor *weight(const std::string &name) const;
    void shape(const std::string &name, std::initializer_list<int64_t> dims) const;
    void linear_shape(const std::string &name, int in, int out, bool bias = true) const;
    void norm_shape(const std::string &name, int dim) const;
    bool f32_activation_compute() const { return source ? source->f32_activation_compute() : f32_activations; }
    void validate_tensor_coverage() const;
private:
    const ModelLoader *source = nullptr;
    std::string tensor_prefix, metadata_prefix;
    bool bundled = false;
    bool mulan = false;
    bool f32_activations = false;
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
