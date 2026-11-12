#pragma once

#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clap {
// Internal runtime configuration shared by GGUF loading and backend selection.
// Empty device uses CLAP_DEVICE, falling back to "auto"; "cpu" forces CPU.
// Storage precision is independent of activation arithmetic.
enum class ActivationPrecision { ModelDefault, F32 };
struct RuntimeOptions {
    int threads = 0;
    std::string device;
    ActivationPrecision activation_precision = ActivationPrecision::ModelDefault;
};
struct DeviceInfo {
    std::string name, description;
    bool gpu;
};
std::vector<DeviceInfo> available_devices();
} // namespace clap

namespace clap::detail {

// Architecture-independent GGUF ownership, mapping, metadata and tensor checks.
// Keep this alive longer than all model architectures and tensor operations.
class ModelLoader {
  public:
    explicit ModelLoader(RuntimeOptions options = {}) : options(std::move(options)) {}
    const RuntimeOptions &runtime_options() const { return source ? source->runtime_options() : options; }
    size_t tensor_count() const { return source ? source->tensor_count() : tensor_index.size(); }
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
    uint32_t metadata_u32(const char *key) const;
    float metadata_f32(const char *key) const;
    bool metadata_bool(const char *key) const;
    std::vector<std::string> metadata_strings(const char *key) const;
    std::vector<int32_t> metadata_ints(const char *key) const;
    ggml_tensor *weight(const std::string &name) const;
    void shape(const std::string &name, std::initializer_list<int64_t> dims) const;
    void linear_shape(const std::string &name, int in, int out, bool bias = true) const;
    void norm_shape(const std::string &name, int dim) const;
    bool f32_activation_compute() const { return source ? source->f32_activation_compute() : f32_activations; }
    void validate_tensor_coverage() const;

  private:
    int64_t metadata_key(const char *key, gguf_type type) const;
    RuntimeOptions options;
    std::unordered_map<std::string, ggml_tensor *> tensor_index;
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
    // buffers here preserves stable tensor identities and model/backend lifetime.
    std::vector<std::vector<float>> promoted_weights;
#ifdef _WIN32
    std::vector<char> storage;
#endif
};
} // namespace clap::detail
