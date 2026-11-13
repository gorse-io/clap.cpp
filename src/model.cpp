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
#include "model.h"
#include "models/models.h"
#include "quant.h"
#include "vocab.h"

#include <ggml-cpu.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {
void load_backends() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_backend_load_all(); });
}
bool is_gpu(ggml_backend_dev_t device) {
    const auto type = ggml_backend_dev_type(device);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}
bool enabled(ggml_backend_dev_t device) {
    const std::string backend = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
    const char *key = backend == "CUDA"     ? "CLAP_CUDA"
                      : backend == "ROCm"   ? "CLAP_ROCM"
                      : backend == "Vulkan" ? "CLAP_VULKAN"
                      : backend == "MTL"    ? "CLAP_METAL"
                                            : nullptr;
    const char *value = key ? std::getenv(key) : nullptr;
    return !value || std::strcmp(value, "0") != 0;
}
} // namespace
namespace clap {
std::vector<DeviceInfo> available_devices() {
    load_backends();
    std::vector<DeviceInfo> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto device = ggml_backend_dev_get(i);
        devices.push_back({ggml_backend_dev_name(device), ggml_backend_dev_description(device), is_gpu(device)});
    }
    return devices;
}
} // namespace clap

namespace clap::detail {
struct ModelWeights::State {
    using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using TensorContext = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    mutable std::mutex mutex;
    Backend gpu{nullptr, ggml_backend_free};
    Backend cpu{nullptr, ggml_backend_free};
    struct Weight {
        TensorContext context{nullptr, ggml_free};
        Buffer buffer{nullptr, ggml_backend_buffer_free};
        ggml_tensor *tensor = nullptr;
    };
    struct Key {
        ggml_tensor *source;
        ggml_backend_dev_t device;
        ggml_backend_buffer_type_t buffer_type;
        int64_t rows;
        bool operator==(const Key &) const = default;
    };
    struct Hash {
        size_t operator()(const Key &key) const noexcept {
            size_t value = std::hash<ggml_tensor *>{}(key.source);
            const auto mix = [&](size_t next) { value ^= next + size_t(0x9e3779b9) + (value << 6) + (value >> 2); };
            mix(std::hash<ggml_backend_dev_t>{}(key.device));
            mix(std::hash<ggml_backend_buffer_type_t>{}(key.buffer_type));
            mix(std::hash<int64_t>{}(key.rows));
            return value;
        }
    };
    std::unordered_map<Key, std::unique_ptr<Weight>, Hash> entries;
    int threads;
    bool vulkan = false, metal = false;
    ModelWeightStats stats;
    explicit State(const ModelLoader &source) : threads(source.runtime_options().threads) {
        std::string requested = source.runtime_options().device;
        if (requested.empty()) {
            const char *environment = std::getenv("CLAP_DEVICE");
            requested = environment && *environment ? environment : "auto";
        }
        // Explicit CPU execution must not initialize GPU drivers or compile shaders.
        if (requested != "cpu" && requested != "CPU")
            load_backends();
        cpu.reset(ggml_backend_cpu_init());
        if (!cpu)
            throw std::runtime_error("CLAP: CPU backend initialization failed");
        if (requested == "auto") {
            for (size_t i = 0; i < ggml_backend_dev_count() && !gpu; ++i) {
                auto device = ggml_backend_dev_get(i);
                if (is_gpu(device) && enabled(device))
                    gpu.reset(ggml_backend_dev_init(device, nullptr));
            }
        } else if (requested != "cpu" && requested != ggml_backend_dev_name(ggml_backend_get_device(cpu.get()))) {
            auto device = ggml_backend_dev_by_name(requested.c_str());
            if (!device)
                throw std::runtime_error("CLAP: unknown GGML device: " + requested + "; use clap devices");
            gpu.reset(ggml_backend_dev_init(device, nullptr));
            if (!gpu)
                throw std::runtime_error("CLAP: cannot initialize requested GGML device: " + requested);
        }
        if (gpu) {
            const std::string backend =
                ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu.get())));
            vulkan = backend == "Vulkan";
            metal = backend == "MTL";
        }
    }
};
ModelWeights::ModelWeights(const ModelLoader &loader) : state(std::make_unique<State>(loader)) {}
ModelWeights::~ModelWeights() = default;
ggml_backend_dev_t ModelWeights::gpu_device() const {
    return state->gpu ? ggml_backend_get_device(state->gpu.get()) : nullptr;
}
bool ModelWeights::uses_vulkan() const { return state->vulkan; }
bool ModelWeights::uses_metal() const { return state->metal; }
int ModelWeights::default_threads() const { return state->threads; }
ModelWeightStats ModelWeights::stats() const {
    std::lock_guard<std::mutex> guard(state->mutex);
    return state->stats;
}
ggml_tensor *ModelWeights::get(ggml_tensor *source, ggml_backend_t compute, bool device, bool &uploaded) {
    uploaded = false;
    std::lock_guard<std::mutex> guard(state->mutex);
    auto allocation = device ? state->gpu.get() : state->cpu.get();
    if (!allocation || !compute)
        throw std::runtime_error("CLAP: shared weight backend is unavailable");
    auto buffer_type = ggml_backend_get_default_buffer_type(allocation);
    if (!device && reinterpret_cast<uintptr_t>(source->data) % ggml_backend_buft_get_alignment(buffer_type) == 0)
        buffer_type = ggml_backend_cpu_buffer_type();
    if (ggml_backend_get_device(compute) != ggml_backend_get_device(allocation) ||
        !ggml_backend_supports_buft(compute, buffer_type))
        throw std::runtime_error("CLAP: execution backend does not support shared model weights");
    const int64_t rows = device && state->vulkan ? (source->ne[1] + 31) / 32 * 32 : source->ne[1];
    const State::Key key{source, ggml_backend_get_device(allocation), buffer_type, rows};
    const auto found = state->entries.find(key);
    if (found != state->entries.end())
        return found->second->tensor;
    if (source->op != GGML_OP_NONE || !*ggml_get_name(source))
        throw std::runtime_error("CLAP: shared model weights must be named leaf tensors");
    const std::string name = ggml_get_name(source);
    auto weight = std::make_unique<State::Weight>();
    weight->context.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    if (!weight->context)
        throw std::runtime_error("CLAP: weight metadata allocation failed");
    auto *tensor = ggml_dup_tensor(weight->context.get(), source);
    ggml_set_name(tensor, name.c_str());
    if (device) {
        // Keep the cooperative-matrix tile workaround for the pinned Vulkan backend.
        if (state->vulkan)
            tensor->ne[1] = (tensor->ne[1] + 31) / 32 * 32;
        tensor->nb[2] = tensor->nb[1] * tensor->ne[1];
        tensor->nb[3] = tensor->nb[2] * tensor->ne[2];
        weight->buffer.reset(ggml_backend_alloc_ctx_tensors(weight->context.get(), state->gpu.get()));
        if (!weight->buffer)
            throw std::runtime_error("CLAP: device weight allocation failed");
        const size_t bytes = ggml_nbytes(source);
        if (ggml_nbytes(tensor) > bytes)
            ggml_backend_tensor_memset(tensor, 0, bytes, ggml_nbytes(tensor) - bytes);
        ggml_backend_tensor_set(tensor, source->data, 0, bytes);
    } else {
        const size_t alignment =
            ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(state->cpu.get()));
        if (reinterpret_cast<uintptr_t>(source->data) % alignment) {
            // Promoted F16 weights can come from std::vector storage with weaker alignment.
            weight->buffer.reset(ggml_backend_alloc_ctx_tensors(weight->context.get(), state->cpu.get()));
            if (!weight->buffer)
                throw std::runtime_error("CLAP: aligned weight allocation failed");
            ggml_backend_tensor_set(tensor, source->data, 0, ggml_nbytes(source));
        } else {
            weight->buffer.reset(ggml_backend_cpu_buffer_from_ptr(source->data, ggml_nbytes(source)));
            if (!weight->buffer ||
                ggml_backend_tensor_alloc(weight->buffer.get(), tensor, source->data) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("CLAP: mapped weight buffer initialization failed");
        }
    }
    // Small affine vectors should follow adjacent compute instead of pinning it to the CPU.
    ggml_backend_buffer_set_usage(weight->buffer.get(), device || !ggml_is_vector(source)
                                                            ? GGML_BACKEND_BUFFER_USAGE_WEIGHTS
                                                            : GGML_BACKEND_BUFFER_USAGE_ANY);
    weight->tensor = tensor;
    auto *result = weight->tensor;
    const size_t bytes = ggml_backend_buffer_get_size(weight->buffer.get());
    state->entries.emplace(key, std::move(weight));
    if (device) {
        ++state->stats.device_tensors;
        state->stats.device_bytes += bytes;
        ++state->stats.uploads;
        uploaded = true;
    } else {
        ++state->stats.host_tensors;
        state->stats.host_bytes += bytes;
    }
    return result;
}
} // namespace clap::detail

clap::detail::Model::Model(const std::string &path, const clap::RuntimeOptions &options) : loader(options) {
    using namespace clap::detail;
    clap::detail::load_model_weights(loader, path);
    const auto name = loader.metadata("general.architecture");
    const auto initialize_weights = [&]() -> const ModelLoader & {
        weights = std::make_shared<ModelWeights>(loader);
        return loader;
    };
    if (name == "clap")
        architecture = std::make_unique<ClapModel>(initialize_weights());
    else if (name == "clap2")
        architecture = std::make_unique<Clap2Model>(initialize_weights());
    else if (name == "clamp3")
        architecture = std::make_unique<Clamp3Model>(initialize_weights());
    else if (name == "mert2")
        architecture = std::make_unique<Mert2Model>(initialize_weights());
    else if (name == "mert")
        architecture = std::make_unique<MertModel>(initialize_weights());
    else if (name == "muq_mulan")
        architecture = std::make_unique<MuqMulanModel>(initialize_weights());
    else if (name == "muq")
        architecture = std::make_unique<MuqModel>(initialize_weights());
    else
        throw std::runtime_error("CLAP: unsupported GGUF architecture: " + name);
    loader.validate_tensor_coverage();
    if (architecture->has_text_encoder()) {
        const auto text = architecture->text_parameters();
        const auto tokenizer = nlohmann::json::parse(loader.metadata((name + ".tokenizer").c_str()));
        if (name == "clamp3" || name == "muq_mulan")
            vocabulary = std::make_unique<UnigramVocabulary>(tokenizer, text.bos, text.eos, 3, text.context_length);
        else if (name == "clap2")
            vocabulary = std::make_unique<ByteBpeVocabulary>(tokenizer, text.bos, text.eos, text.context_length, true);
        else
            vocabulary = std::make_unique<ByteBpeVocabulary>(tokenizer, text.bos, text.eos, text.context_length);
    }
}
clap::detail::Model::~Model() = default;

namespace clap::detail {
namespace {
TensorLoadPolicy select_loading_policy(const ModelLoader &loader) {
    std::string architecture;
    try {
        architecture = loader.metadata("general.architecture");
    } catch (const std::runtime_error &) {
        // Missing or non-string architecture metadata is neutral at this layer.
        // The model facade retains ownership of full architecture validation.
        return {};
    }
    if (architecture == "clap")
        return ClapModel::loading_policy(loader);
    if (architecture == "clap2")
        return Clap2Model::loading_policy(loader);
    if (architecture == "clamp3")
        return Clamp3Model::loading_policy(loader);
    if (architecture == "mert")
        return MertModel::loading_policy(loader);
    if (architecture == "mert2")
        return Mert2Model::loading_policy(loader);
    if (architecture == "muq")
        return MuqModel::loading_policy(loader);
    if (architecture == "muq_mulan")
        return MuqMulanModel::loading_policy(loader);
    return {};
}
} // namespace

void load_model_weights(ModelLoader &loader, const std::string &path, bool inference) {
    loader.load(path, inference, select_loading_policy);
}

TensorLoadPolicy encoder_loading_policy(void (*validate_type)(const ggml_tensor *)) {
    TensorLoadPolicy policy;
    policy.f32_activations = true;
    policy.unrecognized_tensor_error = "CLAP: unrecognized model tensor: ";
    policy.validate_tensor_type = validate_type;
    return policy;
}

void validate_encoder_storage(const ggml_tensor *w, bool quantizable) {
    if (ggml_is_quantized(w->type) && !quantizable)
        throw std::runtime_error("CLAP: quantized non-dense/embedding tensor is unsupported: " + std::string(w->name));
}

bool encoder_quantizable(const ggml_tensor *w, const std::string &architecture) {
    if (architecture == "mert")
        return MertModel::tensor_quantizable(w);
    if (architecture == "mert2")
        return Mert2Model::tensor_quantizable(w);
    if (architecture == "muq")
        return MuqModel::tensor_quantizable(w);
    if (architecture == "muq_mulan")
        return MuqMulanModel::tensor_quantizable(w);
    return false;
}

LayerWeights create_layer(const ModelLoader &loader, const std::string &name,
                          std::initializer_list<int64_t> weight_dims, std::initializer_list<int64_t> bias_dims) {
    auto *weight = loader.create_tensor(name + ".weight", weight_dims);
    auto *bias = bias_dims.size() ? loader.create_tensor(name + ".bias", bias_dims) : nullptr;
    return {weight, bias};
}
LayerWeights create_linear(const ModelLoader &loader, const std::string &name, int64_t input, int64_t output,
                           bool bias) {
    return bias ? create_layer(loader, name, {input, output}, {output}) : create_layer(loader, name, {input, output});
}
LayerWeights create_norm(const ModelLoader &loader, const std::string &name, int64_t width) {
    return create_layer(loader, name, {width}, {width});
}
EmbeddingWeights create_embeddings(const ModelLoader &loader, const std::string &prefix, int64_t hidden, int64_t vocab,
                                   int64_t positions, int64_t types, bool word) {
    return {word ? loader.create_tensor(prefix + ".word_embeddings.weight", {hidden, vocab}) : nullptr,
            loader.create_tensor(prefix + ".position_embeddings.weight", {hidden, positions}),
            loader.create_tensor(prefix + ".token_type_embeddings.weight", {hidden, types}),
            create_norm(loader, prefix + ".LayerNorm", hidden)};
}
TransformerWeights create_bert_layer(const ModelLoader &loader, const std::string &prefix, int64_t hidden,
                                     int64_t intermediate) {
    TransformerWeights layer;
    layer.attention = {create_linear(loader, prefix + ".attention.self.query", hidden, hidden),
                       create_linear(loader, prefix + ".attention.self.key", hidden, hidden),
                       create_linear(loader, prefix + ".attention.self.value", hidden, hidden),
                       create_linear(loader, prefix + ".attention.output.dense", hidden, hidden)};
    layer.attention_norm = create_norm(loader, prefix + ".attention.output.LayerNorm", hidden);
    layer.intermediate = create_linear(loader, prefix + ".intermediate.dense", hidden, intermediate);
    layer.output = create_linear(loader, prefix + ".output.dense", intermediate, hidden);
    layer.output_norm = create_norm(loader, prefix + ".output.LayerNorm", hidden);
    return layer;
}
BatchNormWeights BatchNormWeights::create(const ModelLoader &loader, const std::string &prefix, int64_t width) {
    return {create_norm(loader, prefix, width), loader.create_tensor(prefix + ".running_mean", {width}),
            loader.create_tensor(prefix + ".running_var", {width})};
}

} // namespace clap::detail
