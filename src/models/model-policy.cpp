#include "model-policy.h"
#include "models.h"
#include "quant.h"

#include <stdexcept>

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
} // namespace clap::detail
