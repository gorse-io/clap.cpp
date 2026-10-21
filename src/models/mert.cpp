#include "models.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace clap::detail {
MertModel::MertModel(const ModelLoader &source, int threads) : loader(source), ops(source, threads) {
    load_arch_hparams();
    load_arch_tensors();
}

void MertModel::load_arch_hparams() {
    config = nlohmann::json::parse(loader.metadata("mert.config"));
    if (!(config.at("model_type") == "mert_model" && config.at("sample_rate") == 24000)) throw std::runtime_error("expected a 24 kHz MERT-v1 model");
    if (!(!config.value("feature_extractor_cqt", false) && !config.value("deepnorm", false) &&
          !config.value("do_stable_layer_norm", false) && config.at("feat_extract_norm") == "group" &&
          config.value("feat_proj_layer_norm", true) && !config.value("conv_pos_batch_norm", false) &&
          config.at("hidden_act") == "gelu" && config.at("feat_extract_activation") == "gelu")) throw std::runtime_error("unsupported MERT-v1 frontend/encoder variant");
    for (const char *key : {"hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                            "num_conv_pos_embeddings", "num_conv_pos_embedding_groups"})
        if (!(config.at(key).is_number_integer() && config.at(key).get<int>() > 0)) throw std::runtime_error(std::string("invalid MERT ") + key);
    const int width = config.at("hidden_size"), heads = config.at("num_attention_heads");
    if (!(width % heads == 0 && width % config.at("num_conv_pos_embedding_groups").get<int>() == 0)) throw std::runtime_error("invalid MERT attention/position groups");
    const auto &dims = config.at("conv_dim"), &kernels = config.at("conv_kernel"), &strides = config.at("conv_stride");
    if (!(dims.is_array() && !dims.empty() && dims.size() == kernels.size() && dims.size() == strides.size())) throw std::runtime_error("invalid MERT convolution geometry");
    if (!(config.value("num_feat_extract_layers", static_cast<int>(dims.size())) == static_cast<int>(dims.size()))) throw std::runtime_error("invalid MERT convolution geometry: num_feat_extract_layers");
    int64_t receptive = 1, jump = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        for (const auto *array : {&dims, &kernels, &strides})
            if (!((*array)[i].is_number_integer() && (*array)[i].get<int>() > 0)) throw std::runtime_error("invalid MERT convolution dimension");
        receptive += (kernels[i].get<int64_t>() - 1) * jump;
        jump *= strides[i].get<int64_t>();
        if (!(receptive <= 24000 && jump <= 24000)) throw std::runtime_error("unsupported MERT convolution receptive field");
    }
    minimum_samples = static_cast<int>(receptive);
    const float epsilon = config.at("layer_norm_eps");
    if (!(std::isfinite(epsilon) && epsilon > 0)) throw std::runtime_error("invalid MERT layer_norm_eps");
}

void MertModel::load_arch_tensors() const {
    int input = 1;
    const auto dims = config.at("conv_dim"), kernels = config.at("conv_kernel");
    for (size_t i = 0; i < dims.size(); ++i) {
        const int output = dims[i];
        const auto prefix = "frontend." + std::to_string(i);
        loader.shape(prefix + ".conv.weight", {kernels[i].get<int>(), input, output});
        if (config.at("conv_bias").get<bool>()) loader.shape(prefix + ".conv.bias", {output});
        if (i == 0) loader.norm_shape(prefix + ".layer_norm", output);
        input = output;
    }
    const int hidden = config.at("hidden_size"), intermediate = config.at("intermediate_size");
    loader.norm_shape("projection.layer_norm", input);
    loader.linear_shape("projection.projection", input, hidden);
    loader.shape("position.weight", {config.at("num_conv_pos_embeddings").get<int>(),
                 hidden / config.at("num_conv_pos_embedding_groups").get<int>(), hidden});
    loader.shape("position.bias", {hidden});
    loader.norm_shape("encoder.layer_norm", hidden);
    for (int i = 0; i < config.at("num_hidden_layers").get<int>(); ++i) {
        const auto prefix = "layers." + std::to_string(i);
        for (const char *part : {"q_proj", "k_proj", "v_proj", "out_proj"})
            loader.linear_shape(prefix + ".attention." + part, hidden, hidden);
        loader.norm_shape(prefix + ".layer_norm", hidden);
        loader.norm_shape(prefix + ".final_layer_norm", hidden);
        loader.linear_shape(prefix + ".ffn.in", hidden, intermediate);
        loader.linear_shape(prefix + ".ffn.out", intermediate, hidden);
    }
}

std::vector<float> MertModel::text(const std::vector<int> &) const {
    throw std::runtime_error("MERT-v1 has no text encoder");
}
std::vector<float> MertModel::audio_features(const std::vector<float> &, int) const {
    throw std::runtime_error("MERT-v1 requires raw waveform input");
}

std::vector<float> MertModel::normalize(const std::vector<float> &pcm) const {
    if (!(pcm.size() >= static_cast<size_t>(minimum_samples) && pcm.size() <= (1ULL << 28))) throw std::runtime_error("MERT waveform is empty, too short or too large");
    double mean = 0, variance = 0;
    for (float v : pcm) { if (!(std::isfinite(v))) throw std::runtime_error("non-finite MERT waveform"); mean += v; }
    mean /= pcm.size();
    for (float v : pcm) variance += (v - mean) * (v - mean);
    const double scale = 1 / std::sqrt(variance / pcm.size() + 1e-7);
    std::vector<float> result(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) result[i] = static_cast<float>((pcm[i] - mean) * scale);
    return result;
}

// Grouped im2col + GGML matrix multiply, bounded to one group's workspace.
// Stored convolution weights [out,in/group,kernel] are already dense in tap order.
Matrix MertModel::convolution(const Matrix &x, const std::string &name, int kernel,
                             int stride, int padding, int groups, bool bias) const {
    const auto *weight = loader.weight(name + ".weight");
    const int outputs = static_cast<int>(weight->ne[2]);
    const int rows = (x.rows + 2 * padding - kernel) / stride + 1;
    if (!(rows > 0)) throw std::runtime_error("MERT convolution produces no frames");
    const int inputs = x.cols / groups, per_group = outputs / groups, width = inputs * kernel;
    Matrix result(rows, outputs);
    // Expand F16 storage to F32 once per group, keeping activations F32 as in
    // the storage-rounded oracle (GGML F16 dot kernels otherwise round inputs).
    const size_t bytes = (static_cast<size_t>(width) * (rows + per_group) +
                          static_cast<size_t>(rows) * per_group) * sizeof(float) + 2 * 1024 * 1024;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes,nullptr,false}),ggml_free);
    if (!(ctx != nullptr)) throw std::runtime_error("MERT convolution allocation failed");
    auto *w = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,width,per_group);
    auto *input = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,width,rows);
    auto *output = ggml_mul_mat(ctx.get(),w,input);
    auto *graph = ggml_new_graph_custom(ctx.get(),32,false);
    ggml_build_forward_expand(graph,output);
    for (int group = 0; group < groups; ++group) {
        for (int i = 0; i < width * per_group; ++i)
            static_cast<float *>(w->data)[i] = scalar(weight,static_cast<size_t>(group)*width*per_group+i);
        auto *columns = static_cast<float *>(input->data);
        for (int row = 0; row < rows; ++row) for (int channel = 0; channel < inputs; ++channel)
            for (int tap = 0; tap < kernel; ++tap) {
                const int source = row * stride + tap - padding;
                columns[static_cast<size_t>(row)*width + channel*kernel+tap] =
                    source >= 0 && source < x.rows ? x[source][group*inputs+channel] : 0;
            }
        if (!(ggml_graph_compute_with_ctx(ctx.get(),graph,ops.threads()) == GGML_STATUS_SUCCESS)) throw std::runtime_error("MERT convolution GGML execution failed");
        const auto *b = bias ? loader.weight(name + ".bias") : nullptr;
        for (int row = 0; row < rows; ++row) for (int col = 0; col < per_group; ++col)
            result[row][group*per_group+col] = static_cast<float *>(output->data)[row*per_group+col] +
                                             (b ? scalar(b,group*per_group+col) : 0);
    }
    return result;
}

// F16 is storage-only: GGML's F16 dot path also rounds activations, which
// compounds through 12 encoder layers. Expand one dense weight at a time.
Matrix MertModel::linear(const Matrix &x, const std::string &name) const {
    const auto *weight = loader.weight(name + ".weight");
    if (weight->type == GGML_TYPE_F32) return ops.linear(x, name);
    if (weight->type != GGML_TYPE_F16)
        throw std::runtime_error("MERT supports F32/F16 weights only");
    Matrix result(x.rows, static_cast<int>(weight->ne[1]));
    const size_t bytes = (x.data.size() + result.data.size() +
                         static_cast<size_t>(x.cols) * result.cols) * sizeof(float) + 2 * 1024 * 1024;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
    if (!ctx) throw std::runtime_error("MERT linear allocation failed");
    auto *w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, x.cols, result.cols);
    ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(weight->data),
                         static_cast<float *>(w->data), static_cast<int64_t>(x.cols) * result.cols);
    auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, x.cols, x.rows);
    std::memcpy(input->data, x.data.data(), x.data.size() * sizeof(float));
    auto *output = ggml_mul_mat(ctx.get(), w, input);
    auto *graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(ctx.get(), graph, ops.threads()) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("MERT linear GGML execution failed");
    std::memcpy(result.data.data(), output->data, result.data.size() * sizeof(float));
    const auto *bias = loader.weight(name + ".bias");
    for (int row = 0; row < result.rows; ++row)
        for (int col = 0; col < result.cols; ++col) result[row][col] += scalar(bias, col);
    return result;
}

Matrix MertModel::attention(const Matrix &x, const std::string &name) const {
    auto q = linear(x,name+".q_proj"), k = linear(x,name+".k_proj"), v = linear(x,name+".v_proj");
    Matrix out(x.rows,x.cols);
    const int heads = config.at("num_attention_heads"), width = x.cols / heads;
    const float scale = 1/std::sqrt(static_cast<float>(width));
    std::vector<float> scores(x.rows);
    for (int head = 0; head < heads; ++head) for (int row = 0; row < x.rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < x.rows; ++col) {
            float score = 0;
            for (int i = 0; i < width; ++i) score += (q[row][head*width+i]*scale)*k[col][head*width+i];
            scores[col] = score; maximum = std::max(maximum,score);
        }
        double sum = 0;
        for (float &score : scores) { score = std::exp(score-maximum); sum += score; }
        for (int col = 0; col < x.rows; ++col) for (int i = 0; i < width; ++i)
            out[row][head*width+i] += static_cast<float>(scores[col]/sum)*v[col][head*width+i];
    }
    return linear(out,name+".out_proj");
}

Matrix MertModel::forward(const std::vector<float> &pcm, std::vector<float> *layer_mean) const {
    Matrix hidden(static_cast<int>(pcm.size()),1); hidden.data = pcm;
    const auto dims = config.at("conv_dim"), kernels = config.at("conv_kernel"), strides = config.at("conv_stride");
    for (size_t layer = 0; layer < dims.size(); ++layer) {
        const auto name = "frontend." + std::to_string(layer);
        hidden = convolution(hidden,name+".conv",kernels[layer],strides[layer],0,1,config.at("conv_bias"));
        if (layer == 0) {
            // GroupNorm(num_groups=channels): normalize each channel over time.
            const auto *w = loader.weight(name+".layer_norm.weight"), *b = loader.weight(name+".layer_norm.bias");
            for (int col = 0; col < hidden.cols; ++col) {
                double mean = 0, variance = 0;
                for (int row = 0; row < hidden.rows; ++row) mean += hidden[row][col];
                mean /= hidden.rows;
                for (int row = 0; row < hidden.rows; ++row) variance += (hidden[row][col]-mean)*(hidden[row][col]-mean);
                const float scale = static_cast<float>(1/std::sqrt(variance/hidden.rows+1e-5));
                for (int row = 0; row < hidden.rows; ++row)
                    hidden[row][col] = (hidden[row][col]-static_cast<float>(mean))*scale*scalar(w,col)+scalar(b,col);
            }
        }
        gelu(hidden);
    }
    const float epsilon = config.at("layer_norm_eps");
    hidden = linear(ops.norm(hidden,"projection.layer_norm",epsilon),"projection.projection");
    const int kernel = config.at("num_conv_pos_embeddings");
    auto position = convolution(hidden,"position",kernel,1,kernel/2,
                                config.at("num_conv_pos_embedding_groups"),true);
    // Even positional kernels create one extra trailing frame (HF SamePadLayer).
    position.rows = hidden.rows; position.data.resize(hidden.data.size()); gelu(position); add(hidden,position);
    hidden = ops.norm(hidden,"encoder.layer_norm",epsilon);
    const int layers = config.at("num_hidden_layers");
    if (layer_mean) layer_mean->assign(hidden.cols,0);
    auto accumulate = [&] {
        if (!layer_mean) return;
        for (int col = 0; col < hidden.cols; ++col) {
            double sum = 0;
            for (int row = 0; row < hidden.rows; ++row) sum += hidden[row][col];
            (*layer_mean)[col] += static_cast<float>(sum/hidden.rows)/(layers+1);
        }
    };
    accumulate();
    for (int layer = 0; layer < layers; ++layer) {
        const auto prefix = "layers." + std::to_string(layer);
        auto value = attention(hidden,prefix+".attention"); add(value,hidden);
        hidden = ops.norm(value,prefix+".layer_norm",epsilon);
        value = linear(hidden,prefix+".ffn.in"); gelu(value);
        value = linear(value,prefix+".ffn.out"); add(value,hidden);
        hidden = ops.norm(value,prefix+".final_layer_norm",epsilon);
        accumulate();
    }
    return hidden;
}

AudioFrames MertModel::audio_frames(const std::vector<float> &pcm) const {
    auto output = forward(normalize(pcm),nullptr);
    return {std::move(output.data),output.rows,output.cols};
}
AudioFrames MertModel::clamp3_features(const std::vector<float> &pcm) const {
    if (!(pcm.size() >= 24000)) throw std::runtime_error("CLaMP3 MERT extraction requires at least one second of audio");
    auto normalized = normalize(pcm);
    AudioFrames result; result.dimension = projection_dimension();
    constexpr size_t window = 5*24000;
    for (size_t start = 0; start < normalized.size(); start += window) {
        const auto count = std::min(window,normalized.size()-start);
        if (count < 24000) break;
        std::vector<float> chunk(normalized.begin()+start,normalized.begin()+start+count), mean;
        forward(chunk,&mean);
        result.values.insert(result.values.end(),mean.begin(),mean.end()); ++result.frames;
    }
    return result;
}
} // namespace clap::detail
