#pragma once
#include <ggml.h>
#include <regex>
#include <string>

namespace clap::detail {
inline bool encoder_architecture(const std::string &architecture) {
    return architecture == "mert" || architecture == "mert2" || architecture == "muq" || architecture == "muq_mulan";
}
// Only native dense/embedding layouts are block quantizable. In particular,
// Conv1d [kernel,in,out], Conv2d, GRN and mel filters must remain floating point,
// even when their leading dimension happens to be divisible by a block size.
inline bool encoder_quantizable(const ggml_tensor *w, const std::string &architecture) {
    if (ggml_n_dims(w) != 2) return false;
    const std::string name(w->name);
    static const std::regex mert(R"((projection\.projection|layers\.[0-9]+\.(attention\.(q_proj|k_proj|v_proj|out_proj)|ffn\.(in|out)))\.weight)");
    static const std::regex mert2(R"((subsampling_module\.[0-9]+\.convnext_layers\.[0-9]+\.pointwise_block\.(1|4)|layers\.[0-9]+\.(attn\.(query_proj|key_proj|value_proj|out_proj)|ffn[12]\.w_[12]))\.weight)");
    static const std::regex muq(R"((conv\.linear|conformer\.layers\.[0-9]+\.(self_attn\.linear_(q|k|v|out)|ffn[12]\.(intermediate_dense|output_dense)))\.weight)");
    static const std::regex joint(R"((text\.embeddings\.(word_embeddings|position_embeddings|token_type_embeddings)|text\.encoder\.layer\.[0-9]+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)|adapter\.layers\.[0-9]+\.(0\.(to_q|to_kv|to_out\.0)|1\.(1|4))|audio_proj|audio_latent|text_proj)\.weight)");
    if (architecture == "mert") return std::regex_match(name, mert);
    if (architecture == "mert2") return std::regex_match(name, mert2);
    if (architecture == "muq" || architecture == "muq_mulan")
        return std::regex_match(name, muq) || (architecture == "muq_mulan" && std::regex_match(name, joint));
    return false;
}
inline bool encoder_sensitive(const std::string &name, const std::string &architecture) {
    const auto ends = [&](const std::string &suffix) {
        return name.size() >= suffix.size() && name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
    };
    if (architecture == "mert") return name == "projection.projection.weight" || ends(".attention.v_proj.weight") || ends(".ffn.out.weight");
    if (architecture == "mert2") return ends(".attn.value_proj.weight") || ends(".ffn1.w_2.weight") || ends(".ffn2.w_2.weight") || ends(".pointwise_block.4.weight");
    return name == "conv.linear.weight" || ends(".self_attn.linear_v.weight") ||
        ends(".ffn1.output_dense.weight") || ends(".ffn2.output_dense.weight") ||
        ends(".attention.self.value.weight") || (ends(".output.dense.weight") && !ends(".attention.output.dense.weight")) ||
        ends(".0.to_kv.weight") || ends(".1.4.weight") ||
        name == "audio_proj.weight" || name == "audio_latent.weight" || name == "text_proj.weight";
}
} // namespace clap::detail
