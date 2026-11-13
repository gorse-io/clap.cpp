/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

enum class TensorRequirement { Required, Optional };

// Supplied by the model layer; the generic loader does not select architectures.
// Validators run at the same per-tensor stages as the storage checks they extend.
struct TensorLoadPolicy {
    bool f32_activations = false;
    const char *unrecognized_tensor_error = nullptr;
    void (*validate_tensor_type)(const ggml_tensor *) = nullptr;
    void (*validate_tensor_data)(const ggml_tensor *) = nullptr;
    bool (*promote_f16)(const ggml_tensor *) = nullptr;
    void (*validate_tensor_contract)(const ggml_tensor *, std::initializer_list<int64_t>) = nullptr;
};
class ModelLoader;
using TensorLoadPolicySelector = TensorLoadPolicy (*)(const ModelLoader &);

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

    // No selector means neutral storage loading. Model callers supply policies
    // explicitly; storage tools disable promotion before conversion/copying.
    void load(const std::string &path, bool inference = true, TensorLoadPolicySelector select_policy = nullptr);
    std::string metadata(const char *key) const;
    ggml_tensor *weight(const std::string &name) const;
    // Resolve and validate an existing mapped tensor, as llama.cpp's
    // create_tensor contract does; this does not allocate model weights.
    ggml_tensor *create_tensor(const std::string &name, std::initializer_list<int64_t> dims,
                               TensorRequirement requirement = TensorRequirement::Required) const;
    bool f32_activation_compute() const { return source ? source->f32_activation_compute() : policy.f32_activations; }
    void validate_tensor_coverage() const;

  private:
    RuntimeOptions options;
    std::unordered_map<std::string, ggml_tensor *> tensor_index;
    const ModelLoader *source = nullptr;
    std::string tensor_prefix, metadata_prefix;
    TensorLoadPolicy policy;
    mutable std::unordered_set<std::string> validated_shapes;
    gguf_context *gguf = nullptr;
    ggml_context *weights = nullptr;
    void *mapping = nullptr;
    size_t mapping_size = 0;
    // Policies can request F32 storage while the owner retains stable tensor
    // identities and the promoted buffers for the complete model lifetime.
    std::vector<std::vector<float>> promoted_weights;
#ifdef _WIN32
    std::vector<char> storage;
#endif
};
} // namespace clap::detail
