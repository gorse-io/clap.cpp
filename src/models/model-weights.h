#pragma once

#include "graph.h"

namespace clap::detail {
// Bound once after GGUF shape validation. Graph construction never resolves names.
struct AttentionWeights {
    LayerWeights query, key, value, output;
    ggml_tensor *relative_bias = nullptr;
};
struct TransformerWeights {
    AttentionWeights attention;
    LayerWeights attention_norm, output_norm, intermediate, output;
};
struct EmbeddingWeights {
    ggml_tensor *word = nullptr, *position = nullptr, *type = nullptr;
    LayerWeights norm;
};
inline EmbeddingWeights bind_embeddings(const ModelLoader &loader, const std::string &prefix, bool word = true) {
    return {word ? loader.weight(prefix + ".word_embeddings.weight") : nullptr,
            loader.weight(prefix + ".position_embeddings.weight"),
            loader.weight(prefix + ".token_type_embeddings.weight"), LayerWeights::bind(loader, prefix + ".LayerNorm")};
}
inline TransformerWeights bind_bert_layer(const ModelLoader &loader, const std::string &prefix) {
    return {{LayerWeights::bind(loader, prefix + ".attention.self.query"),
             LayerWeights::bind(loader, prefix + ".attention.self.key"),
             LayerWeights::bind(loader, prefix + ".attention.self.value"),
             LayerWeights::bind(loader, prefix + ".attention.output.dense")},
            LayerWeights::bind(loader, prefix + ".attention.output.LayerNorm"),
            LayerWeights::bind(loader, prefix + ".output.LayerNorm"),
            LayerWeights::bind(loader, prefix + ".intermediate.dense"),
            LayerWeights::bind(loader, prefix + ".output.dense")};
}
struct BatchNormWeights {
    LayerWeights affine;
    ggml_tensor *mean = nullptr, *variance = nullptr;
    static BatchNormWeights bind(const ModelLoader &loader, const std::string &prefix) {
        return {LayerWeights::bind(loader, prefix), loader.weight(prefix + ".running_mean"),
                loader.weight(prefix + ".running_var")};
    }
};
struct FeedForwardWeights {
    LayerWeights norm, input, output;
};
struct ConvolutionWeights {
    LayerWeights norm, input, depthwise, output_norm, output;
    BatchNormWeights batch_norm;
};
struct ConformerWeights {
    FeedForwardWeights ffn1, ffn2;
    LayerWeights attention_norm, final_norm;
    AttentionWeights attention;
    ConvolutionWeights convolution;
};
} // namespace clap::detail
