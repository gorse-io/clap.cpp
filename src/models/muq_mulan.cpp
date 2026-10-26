#include "models.h"
#include "model-context.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace clap::detail {
namespace {
void positive(const nlohmann::json &obj, const char *key) {
    if (!obj.at(key).is_number_integer() || obj.at(key).get<int64_t>() <= 0 || obj.at(key).get<int64_t>() > 10000000)
        throw std::runtime_error(std::string("CLAP: invalid MuLan dimension: ") + key);
}
}
// Execution is kept separate from the validated model/tensor contract below.
struct MuqMulanModel::context final : ModelContext {
    context(const MuqMulanModel &model, int threads)
        : ModelContext(model.loader, threads),
          model(model),
          config(model.config),
          text_config(model.text_config),
          audio_backbone(model.audio_backbone) {}

    int projection_dimension() const { return model.projection_dimension(); }
    Matrix encode_text(const Matrix &x) const;
    Matrix adapter(Matrix x) const;
    Matrix biasless_norm(const Matrix &x, const std::string &name) const;
    std::vector<float> project(const Matrix &x, const std::string &name) const;
    std::vector<float> text(const std::vector<int> &ids) const override;
    AudioFrames audio_frames(const std::vector<float> &) const override;
    std::vector<float> waveform_embedding(const std::vector<float> &pcm, size_t crop_offset = 0) const override;

private:
    const MuqMulanModel &model;
    const nlohmann::json &config;
    const nlohmann::json &text_config;
    const MuqModel &audio_backbone;
};

std::unique_ptr<ModelContext> MuqMulanModel::build_arch_context(int threads) const {
    return std::make_unique<context>(*this, threads);
}

MuqMulanModel::MuqMulanModel(const ModelLoader &source, int threads)
    : loader(source), audio_backbone(source, threads) {
    load_arch_hparams(); load_arch_tensors();
    inference = build_arch_context(threads);
}
MuqMulanModel::~MuqMulanModel() = default;

void MuqMulanModel::load_arch_hparams() {
    config = nlohmann::json::parse(loader.metadata("muq_mulan.config"));
    text_config = nlohmann::json::parse(loader.metadata("muq_mulan.text_config"));
    const auto &m = config.at("mulan"), &a = config.at("audio_transformer"), &t = config.at("text_transformer");
    positive(m,"clip_secs"); positive(m,"dim_latent");
    if (m.at("clip_secs").get<int64_t>() > 3600 || !m.at("sr").is_number_integer() || m.at("sr") != 24000 || m.value("hierarchical_contrastive_loss",false) ||
        config.at("audio_model").at("name") != "OpenMuQ/MuQ-large-msd-iter" ||
        !config.at("audio_model").at("use_layer_idx").is_number_integer() ||
        config.at("audio_model").at("use_layer_idx") != -1 ||
        !config.at("audio_model").at("model_dim").is_number_integer() ||
        config.at("audio_model").at("model_dim") != audio_backbone.projection_dimension() || a.at("tf_depth") != 0 ||
        config.at("text_model").at("name") != "xlm-roberta-base")
        throw std::runtime_error("CLAP: unsupported MuLan backbone/layer/clip variant");
    for (const auto *branch : {&a,&t}) {
        for (const auto *key : {"dim","heads","dim_head"}) positive(*branch,key);
        const double mult = branch->at("ff_mult").get<double>();
        const int64_t inner = branch->at("heads").get<int64_t>() * branch->at("dim_head").get<int64_t>();
        const double ff = branch->at("dim").get<double>() * mult * 2 / 3;
        if (!(inner <= 1048576 && std::isfinite(ff) && ff >= 1 && ff <= 1048576 &&
              std::isfinite(mult) && mult > 0 && mult <= 100 &&
              branch->at("tf_depth").is_number_integer() && branch->at("tf_depth").get<int64_t>() >= 0 && branch->at("tf_depth").get<int64_t>() <= 1000))
            throw std::runtime_error("CLAP: invalid MuLan adapter dimensions");
    }
    for (const auto *key : {"hidden_size","num_hidden_layers","num_attention_heads","intermediate_size","vocab_size","max_position_embeddings","type_vocab_size"}) positive(text_config,key);
    const float eps=text_config.at("layer_norm_eps");
    if (t.at("dim") != text_config.at("hidden_size") ||
        !(config.at("text_model").at("model_dim").is_null() || (config.at("text_model").at("model_dim").is_number_integer() && config.at("text_model").at("model_dim") == text_config.at("hidden_size"))) ||
        text_config.at("model_type") != "xlm-roberta" || text_config.at("hidden_act") != "gelu" ||
        text_config.value("position_embedding_type",std::string("absolute")) != "absolute" ||
        text_config.value("is_decoder",false) || text_config.value("add_cross_attention",false) ||
        text_config.at("hidden_size").get<int>() % text_config.at("num_attention_heads").get<int>() != 0 ||
        !text_config.at("pad_token_id").is_number_integer() || !text_config.at("bos_token_id").is_number_integer() ||
        !text_config.at("eos_token_id").is_number_integer() || text_config.at("pad_token_id") != 1 || text_config.at("bos_token_id") != 0 || text_config.at("eos_token_id") != 2 ||
        text_config.at("max_position_embeddings").get<int>() < 4 || text_config.at("vocab_size").get<int>() < 4 ||
        !(std::isfinite(eps) && eps > 0))
        throw std::runtime_error("CLAP: unsupported MuLan XLM-R configuration");
    const auto tokenizer=nlohmann::json::parse(loader.metadata("muq_mulan.tokenizer"));
    const auto &vocab=tokenizer.at("model").at("vocab");
    if (tokenizer.at("model").value("type",std::string("Unigram")) != "Unigram" || vocab.size() != text_config.at("vocab_size").get<size_t>() ||
        vocab.at(0).at(0) != "<s>" || vocab.at(1).at(0) != "<pad>" || vocab.at(2).at(0) != "</s>" || vocab.at(3).at(0) != "<unk>")
        throw std::runtime_error("CLAP: MuLan tokenizer/config mismatch");
}
void MuqMulanModel::load_arch_tensors() const {
    const int d=text_config.at("hidden_size"), dim=config.at("text_transformer").at("dim");
    loader.shape("text.embeddings.word_embeddings.weight",{d,text_config.at("vocab_size").get<int>()});
    loader.shape("text.embeddings.position_embeddings.weight",{d,text_config.at("max_position_embeddings").get<int>()});
    loader.shape("text.embeddings.token_type_embeddings.weight",{d,text_config.at("type_vocab_size").get<int>()});
    loader.norm_shape("text.embeddings.LayerNorm",d);
    for(int i=0;i<text_config.at("num_hidden_layers").get<int>();++i) {
        const auto p="text.encoder.layer."+std::to_string(i);
        for(const auto *name:{"query","key","value"}) loader.linear_shape(p+".attention.self."+name,d,d);
        loader.linear_shape(p+".attention.output.dense",d,d); loader.norm_shape(p+".attention.output.LayerNorm",d);
        loader.linear_shape(p+".intermediate.dense",d,text_config.at("intermediate_size"));
        loader.linear_shape(p+".output.dense",text_config.at("intermediate_size"),d); loader.norm_shape(p+".output.LayerNorm",d);
    }
    const auto &t=config.at("text_transformer");
    const int inner=t.at("heads").get<int>()*t.at("dim_head").get<int>();
    const int ff=static_cast<int>(dim*t.at("ff_mult").get<double>()*2/3);
    if (ff<=0) throw std::runtime_error("CLAP: invalid MuLan GEGLU width");
    for(int i=0;i<t.at("tf_depth").get<int>();++i) {
        const auto p="adapter.layers."+std::to_string(i);
        loader.shape(p+".0.norm.learned_gamma",{dim});
        loader.shape(p+".0.q_scale",{t.at("dim_head").get<int>()}); loader.shape(p+".0.k_scale",{t.at("dim_head").get<int>()});
        loader.linear_shape(p+".0.to_q",dim,inner,false); loader.linear_shape(p+".0.to_kv",dim,2*inner,false);
        loader.linear_shape(p+".0.to_out.0",inner,dim,false);
        loader.shape(p+".1.0.learned_gamma",{dim});
        loader.linear_shape(p+".1.1",dim,2*ff,false); loader.linear_shape(p+".1.4",ff,dim,false);
    }
    loader.linear_shape("audio_proj",audio_backbone.projection_dimension(),config.at("audio_transformer").at("dim"));
    loader.linear_shape("audio_latent",config.at("audio_transformer").at("dim"),projection_dimension());
    loader.linear_shape("text_proj",dim,projection_dimension());
}
TextParameters MuqMulanModel::text_parameters() const {
    // Tokenize without truncation; reject over-context input in text(), since
    // the official wrapper also does not truncate or chunk a single item.
    return {0,2,std::numeric_limits<size_t>::max()};
}
int MuqMulanModel::projection_dimension() const { return config.at("mulan").at("dim_latent"); }
AudioFrames MuqMulanModel::context::audio_frames(const std::vector<float> &) const { throw std::runtime_error("CLAP: MuLan exposes embeddings, not frames"); }
Matrix MuqMulanModel::context::encode_text(const Matrix &embeddings) const {
    const int heads=text_config.at("num_attention_heads"), hd=embeddings.cols/heads;
    const float eps=text_config.at("layer_norm_eps");
    auto x=ops.norm(embeddings,"text.embeddings.LayerNorm",eps);
    for(int layer=0;layer<text_config.at("num_hidden_layers").get<int>();++layer) {
        const auto p="text.encoder.layer."+std::to_string(layer);
        auto q=ops.linear(x,p+".attention.self.query"), k=ops.linear(x,p+".attention.self.key"), v=ops.linear(x,p+".attention.self.value");
        Matrix y(x.rows,x.cols); std::vector<float> scores(x.rows);
        for(int h=0;h<heads;++h) for(int i=0;i<x.rows;++i) {
            float maximum=-std::numeric_limits<float>::infinity();
            for(int j=0;j<x.rows;++j) {
                float sum=0; for(int d=0;d<hd;++d) sum+=q[i][h*hd+d]*k[j][h*hd+d];
                scores[j]=sum/std::sqrt(static_cast<float>(hd)); maximum=std::max(maximum,scores[j]);
            }
            float sum=0; for(float &s:scores) {s=std::exp(s-maximum);sum+=s;}
            for(int j=0;j<x.rows;++j) for(int d=0;d<hd;++d) y[i][h*hd+d]+=scores[j]/sum*v[j][h*hd+d];
        }
        y=ops.linear(y,p+".attention.output.dense");add(y,x);x=ops.norm(y,p+".attention.output.LayerNorm",eps);
        y=ops.linear(x,p+".intermediate.dense");gelu(y);y=ops.linear(y,p+".output.dense");add(y,x);
        x=ops.norm(y,p+".output.LayerNorm",eps);
    }
    return x;
}
Matrix MuqMulanModel::context::biasless_norm(const Matrix &x,const std::string &name) const {
    Matrix y(x.rows,x.cols); const auto *gamma=loader.weight(name+".learned_gamma");
    for(int r=0;r<x.rows;++r) {
        double mean=0,var=0; for(int c=0;c<x.cols;++c) mean+=x[r][c]; mean/=x.cols;
        for(int c=0;c<x.cols;++c) {double d=x[r][c]-mean;var+=d*d;}
        const float scale=1/std::sqrt(var/x.cols+1e-5);
        for(int c=0;c<x.cols;++c) y[r][c]=(x[r][c]-static_cast<float>(mean))*scale*scalar(gamma,c);
    }
    return y;
}
Matrix MuqMulanModel::context::adapter(Matrix x) const {
    const auto &t=config.at("text_transformer"); const int heads=t.at("heads"),hd=t.at("dim_head"),inner=heads*hd;
    for(int layer=0;layer<t.at("tf_depth").get<int>();++layer) {
        const auto p="adapter.layers."+std::to_string(layer);
        auto normalized=biasless_norm(x,p+".0.norm");
        auto q=ops.linear(normalized,p+".0.to_q",false), kv=ops.linear(normalized,p+".0.to_kv",false);
        const auto *qs=loader.weight(p+".0.q_scale"),*ks=loader.weight(p+".0.k_scale");
        for(int r=0;r<x.rows;++r) for(int h=0;h<heads;++h) {
            float qnorm=0,knorm=0;
            for(int d=0;d<hd;++d) {qnorm+=q[r][h*hd+d]*q[r][h*hd+d];knorm+=kv[r][h*hd+d]*kv[r][h*hd+d];}
            qnorm=std::max(std::sqrt(qnorm),1e-12f);knorm=std::max(std::sqrt(knorm),1e-12f);
            for(int d=0;d<hd;++d) {q[r][h*hd+d]=q[r][h*hd+d]/qnorm*scalar(qs,d);kv[r][h*hd+d]=kv[r][h*hd+d]/knorm*scalar(ks,d);}
        }
        Matrix attended(x.rows,inner);std::vector<float> scores(x.rows);
        for(int h=0;h<heads;++h) for(int i=0;i<x.rows;++i) {
            float maximum=-std::numeric_limits<float>::infinity();
            for(int j=0;j<x.rows;++j) {float sum=0;for(int d=0;d<hd;++d) sum+=q[i][h*hd+d]*kv[j][h*hd+d];scores[j]=sum*8;maximum=std::max(maximum,scores[j]);}
            float sum=0;for(float &s:scores) {s=std::exp(s-maximum);sum+=s;}
            for(int j=0;j<x.rows;++j) for(int d=0;d<hd;++d) attended[i][h*hd+d]+=scores[j]/sum*kv[j][inner+h*hd+d];
        }
        auto y=ops.linear(attended,p+".0.to_out.0",false);add(x,y);
        y=ops.linear(biasless_norm(x,p+".1.0"),p+".1.1",false);
        const int ff=y.cols/2;Matrix gate(y.rows,ff),value(y.rows,ff);
        for(int r=0;r<y.rows;++r) for(int c=0;c<ff;++c) {value[r][c]=y[r][c];gate[r][c]=y[r][ff+c];}
        gelu(gate);for(size_t i=0;i<value.data.size();++i) value.data[i]*=gate.data[i];
        y=ops.linear(value,p+".1.4",false);add(x,y);
    }
    return x;
}
std::vector<float> MuqMulanModel::context::project(const Matrix &x,const std::string &name) const {
    Matrix pooled(1,x.cols);for(int r=0;r<x.rows;++r) for(int c=0;c<x.cols;++c) pooled[0][c]+=x[r][c]/x.rows;
    auto result=ops.linear(pooled,name).data;
    float norm=0;for(float v:result) norm+=v*v;norm=std::max(std::sqrt(norm),1e-12f);
    for(float &v:result) {v/=norm;if(!std::isfinite(v)) throw std::runtime_error("CLAP: non-finite MuLan embedding");}
    return result;
}
std::vector<float> MuqMulanModel::context::text(const std::vector<int> &ids) const {
    const int dim=text_config.at("hidden_size"),positions=text_config.at("max_position_embeddings"),vocab=text_config.at("vocab_size");
    if(ids.empty() || ids.size()>static_cast<size_t>(positions-2)) throw std::runtime_error("CLAP: MuLan text is empty or exceeds XLM-R context; truncation/chunking unsupported");
    Matrix x(static_cast<int>(ids.size()),dim);std::vector<float> pos(dim),type(dim);
    tensor_row(loader.weight("text.embeddings.token_type_embeddings.weight"),0,type.data());
    int position=1;
    for(size_t r=0;r<ids.size();++r) {
        const int id=ids[r];if(id<0 || id>=vocab) throw std::runtime_error("CLAP: MuLan token ID out of vocabulary");
        const int p=id==1?1:++position;
        if(p>=positions) throw std::runtime_error("CLAP: MuLan text positions exceed context");
        tensor_row(loader.weight("text.embeddings.word_embeddings.weight"),id,x[static_cast<int>(r)]);
        tensor_row(loader.weight("text.embeddings.position_embeddings.weight"),p,pos.data());
        for(int c=0;c<dim;++c) x[static_cast<int>(r)][c]+=pos[c]+type[c];
    }
    return project(adapter(encode_text(x)),"text_proj");
}
std::vector<float> MuqMulanModel::context::waveform_embedding(const std::vector<float> &pcm, size_t crop_offset) const {
    if (crop_offset != 0) throw std::runtime_error("CLAP: crop offsets are unsupported for this model");
    if(pcm.empty() || pcm.size()>(size_t{1}<<28) || !std::all_of(pcm.begin(),pcm.end(),[](float v){return std::isfinite(v);}))
        throw std::runtime_error("CLAP: MuLan requires finite nonempty waveform of at most 1 GiB");
    const size_t delta=24000*config.at("mulan").at("clip_secs").get<size_t>();
    std::vector<float> result(projection_dimension());size_t count=0;
    for(size_t start=0;start<pcm.size();start+=delta) {
        const size_t remaining=std::min(delta,pcm.size()-start);
        std::vector<float> clip(pcm.begin()+start,pcm.begin()+start+remaining);
        // Exactly one prefix slice, even when it cannot fill a ten-second clip.
        if(remaining<delta) clip.insert(clip.end(),pcm.begin(),pcm.begin()+std::min(delta-remaining,pcm.size()));
        auto frames=audio_backbone.audio_frames(clip);
        Matrix hidden(frames.frames,frames.dimension);hidden.data=std::move(frames.values);
        const auto vector=project(ops.linear(hidden,"audio_proj"),"audio_latent");
        for(size_t d=0;d<result.size();++d) result[d]+=vector[d];
        ++count;
    }
    for(float &v:result) v/=count;
    return result;
}

// Model-facing entry points preserve the existing internal/public facade.
std::vector<float> MuqMulanModel::text(const std::vector<int> &ids) const {
    return inference->text(ids);
}
AudioFrames MuqMulanModel::audio_frames(const std::vector<float> &pcm) const {
    return inference->audio_frames(pcm);
}
std::vector<float> MuqMulanModel::waveform_embedding(const std::vector<float> &pcm, size_t crop_offset) const {
    return inference->waveform_embedding(pcm, crop_offset);
}
} // namespace clap::detail
