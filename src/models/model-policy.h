#pragma once

#include "model-loader.h"

namespace clap::detail {
void load_model_weights(ModelLoader &, const std::string &, bool inference = true);
TensorLoadPolicy encoder_loading_policy(void (*validate_type)(const ggml_tensor *));
void validate_encoder_storage(const ggml_tensor *, bool quantizable);
} // namespace clap::detail
