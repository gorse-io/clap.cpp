// Gemma4-audio Conformer contract: Hugging Face modeling_gemma4.py and
// llama.cpp tools/mtmd/models/gemma4a.cpp. Audio files/rate conversion belong
// exclusively to src/audio.cpp; this module accepts mono 16 kHz PCM.
#include "embeddinggemma2-audio.h"
#include "clap.h"
#include <ggml-cpu.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace clap::detail {
namespace {
constexpr double pi = 3.14159265358979323846;
constexpr int mel_bins = 128, max_samples = 480000, left_context = 12;
void require(bool valid, const std::string &message) {
    if (!valid) throw std::runtime_error("CLAP: Gemma audio " + message);
}
void silu(Matrix &x) { for (auto &v : x.data) v = v / (1.0f + std::exp(-v)); }
void clip(Matrix &x, float lo, float hi) { for (auto &v : x.data) v = std::clamp(v, lo, hi); }
void finite(const Matrix &x) {
    for (float v : x.data) require(std::isfinite(v), "non-finite activation");
}
// F32 im2col operand and decoded small convolution kernel. Unlike ggml_conv_2d,
// this does not implicitly pack the activation into F16 before convolution.
Matrix convolve(const Matrix &patches, const ggml_tensor *weight, int outputs, int threads) {
    Matrix result(patches.rows, outputs);
    const size_t count = static_cast<size_t>(patches.cols) * outputs;
    const size_t bytes = (patches.data.size() + result.data.size() + count) * sizeof(float) + 2*1024*1024;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes,nullptr,false}),ggml_free);
    require(bool(ctx), "convolution allocation failed");
    auto *input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,patches.cols,patches.rows);
    auto *kernel=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,patches.cols,outputs);
    std::memcpy(input->data,patches.data.data(),patches.data.size()*sizeof(float));
    auto *decoded=static_cast<float *>(kernel->data);
    for (size_t i=0;i<count;++i) decoded[i]=scalar(weight,i);
    auto *output=ggml_mul_mat(ctx.get(),kernel,input);
    auto *graph=ggml_new_graph_custom(ctx.get(),32,false);
    ggml_build_forward_expand(graph,output);
    auto plan=ggml_graph_plan(graph,threads,nullptr);
    std::vector<uint8_t> workspace(plan.work_size); plan.work_data=workspace.data();
    require(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS,"convolution compute failed");
    std::memcpy(result.data.data(),output->data,result.data.size()*sizeof(float));
    return result;
}
} // namespace

GemmaAudioEncoder::GemmaAudioEncoder(const ModelLoader &source,int threads)
    : loader(source), hidden(0), heads(0), layers(0), feedforward(0), channels{0,0}, output(0),
      epsilon(0), ops(source,threads) {
    require(loader.metadata("general.architecture")=="clip","requires clip mmproj architecture");
    require(loader.boolean("clip.has_audio_encoder"),"encoder missing");
    require(loader.metadata("clip.audio.projector_type")=="gemma4a","unsupported projector variant");
    auto dimension=[&](const char *key,int max) {
        const auto n=loader.integer(key);
        require(n>0 && n<=max,std::string("invalid dimension: ")+key);
        return static_cast<int>(n);
    };
    hidden=dimension("clip.audio.embedding_length",4096);
    heads=dimension("clip.audio.attention.head_count",128);
    layers=dimension("clip.audio.block_count",128);
    feedforward=dimension("clip.audio.feed_forward_length",16384);
    require(hidden%heads==0 && hidden%2==0,"incompatible head dimensions");
    require(feedforward==4*hidden,"incompatible feed-forward dimension");
    require(loader.integer("clip.audio.num_mel_bins")==mel_bins,"requires 128 mel bins");
    const int projection=dimension("clip.audio.projection_dim",16384);
    epsilon=static_cast<float>(loader.number("clip.audio.attention.layer_norm_epsilon"));
    require(std::isfinite(epsilon) && epsilon>0,"invalid norm epsilon");
    for(int i=0;i<2;++i) {
        const auto name="a.conv1d."+std::to_string(i);
        const auto *w=loader.weight(name+".weight");
        require(w->ne[3]>0 && w->ne[3]<=4096,"invalid convolution channels");
        channels[i]=static_cast<int>(w->ne[3]);
        loader.shape(name+".weight",{3,3,i?channels[0]:1,channels[i]});
        loader.shape(name+".norm.weight",{channels[i]});
        require(w->type==GGML_TYPE_F32 || w->type==GGML_TYPE_F16 || w->type==GGML_TYPE_BF16,
                "subsampling kernels must be unquantized");
    }
    const auto *out=loader.weight("a.pre_encode.out.weight");
    require(out->ne[1]>0 && out->ne[1]<=8192,"invalid output dimension");
    output=static_cast<int>(out->ne[1]);
    loader.linear_shape("a.input_projection",32*channels[1],hidden,false);
    loader.linear_shape("a.pre_encode.out",hidden,output,true);
    loader.linear_shape("mm.a.input_projection",output,projection,false);
    if(loader.has_tensor_prefix("mm.a.soft_emb_norm.weight")) loader.shape("mm.a.soft_emb_norm.weight",{output});
    const auto norm=[&](const std::string &name,int n) {
        loader.shape(name+".weight",{n});
        auto *w=loader.weight(name+".weight");
        require(!ggml_is_quantized(w->type),"norm vectors must be unquantized: "+name);
        for(int i=0;i<n;++i) require(std::isfinite(scalar(w,i)),"non-finite norm: "+name);
    };
    for(int il=0;il<layers;++il) {
        const auto p="a.blk."+std::to_string(il)+".";
        for(const auto *name:{"ffn_norm","ffn_post_norm","ffn_norm_1","ffn_post_norm_1", "norm_conv",
                "conv_norm","ln2","attn_pre_norm","attn_post_norm"}) norm(p+name,hidden);
        loader.shape(p+"conv_dw.weight",{5,hidden});
        require(!ggml_is_quantized(loader.weight(p+"conv_dw.weight")->type),"depthwise kernel must be unquantized");
        loader.shape(p+"per_dim_scale.weight",{hidden/heads});
        for(int d=0;d<hidden/heads;++d) {
            const float value=scalar(loader.weight(p+"per_dim_scale.weight"),d);
            require(std::isfinite(value) && value>=0,"invalid converted per-dimension scale");
        }
        loader.linear_shape(p+"attn_k_rel",hidden,hidden,false);
        for(const auto *name:{"ffn_up","ffn_down","ffn_up_1","ffn_down_1", "attn_q","attn_k",
                "attn_v","attn_out","conv_pw1","conv_pw2"}) {
            const std::string n=name;
            const bool up=n=="ffn_up" || n=="ffn_up_1";
            const bool down=n=="ffn_down" || n=="ffn_down_1";
            loader.linear_shape(p+n,down?feedforward:hidden,up?feedforward:n=="conv_pw1"?2*hidden:hidden,false);
            for(const auto *suffix:{"input_min","input_max","output_min","output_max"}) {
                loader.shape(p+n+"."+suffix,{1});
                require(std::isfinite(scalar(loader.weight(p+n+"."+suffix),0)),"non-finite clamp: "+p+n);
            }
            for(const auto *kind:{"input","output"}) {
                const auto base=p+n+"."+kind;
                require(scalar(loader.weight(base+"_min"),0)<=scalar(loader.weight(base+"_max"),0),"inverted clamp: "+p+n);
            }
        }
    }
    // The published file also carries vision weights, which are out of scope.
    // Reject unknown audio tensors without requiring coverage of that tower.
    loader.validate_tensor_coverage("a.");
    loader.validate_tensor_coverage("mm.a.");
}

Matrix GemmaAudioEncoder::log_mel(const std::vector<float> &pcm) {
    require(pcm.size()>160,"input is too short (at least 161 samples required)");
    for(float v:pcm) require(std::isfinite(v),"non-finite PCM input");
    const int samples=static_cast<int>(std::min(pcm.size(),static_cast<size_t>(max_samples)));
    const int frames=(samples-161)/160+1;
    Matrix result(frames,mel_bins);
    std::array<double,130> edges;
    for(int i=0;i<130;++i) edges[i]=700.0*(std::pow(10.0,std::log10(1.0+8000.0/700.0)*i/129.0)-1.0);
    std::array<float,320> window;
    for(int n=0;n<320;++n) window[n]=static_cast<float>(0.5-0.5*std::cos(2*pi*n/320));
    std::array<std::array<double,257>,mel_bins> filters{};
    for(int m=0;m<mel_bins;++m) for(int k=0;k<257;++k) {
        const double hz=k*16000.0/512;
        filters[m][k]=std::max(0.0,std::min((hz-edges[m])/(edges[m+1]-edges[m]),
                                         (edges[m+2]-hz)/(edges[m+2]-edges[m+1])));
    }
    std::vector<std::complex<double>> spectrum(512);
    for(int t=0;t<frames;++t) {
        std::fill(spectrum.begin(),spectrum.end(),std::complex<double>{});
        for(int n=0;n<320;++n) {
            const int i=t*160+n-160;
            const float sample=i<0 || i>=samples?0.0f:pcm[i]*window[n];
            spectrum[n]=sample;
        }
        clap::fft(spectrum);
        std::array<double,257> magnitude;
        for(int k=0;k<257;++k) magnitude[k]=std::abs(spectrum[k]);
        for(int m=0;m<mel_bins;++m) {
            double mel=0;
            for(int k=0;k<257;++k) mel+=magnitude[k]*filters[m][k];
            result[t][m]=static_cast<float>(std::log(mel+0.001));
        }
    }
    return result;
}

Matrix GemmaAudioEncoder::rms(const Matrix &x,const std::string &name) const {
    Matrix y(x.rows,x.cols);
    const auto *weight=name.empty()?nullptr:loader.weight(name+".weight");
    for(int r=0;r<x.rows;++r) {
        double sum=0;
        for(int d=0;d<x.cols;++d) sum+=double(x[r][d])*x[r][d];
        const float scale=static_cast<float>(1.0/std::sqrt(sum/x.cols+epsilon));
        for(int d=0;d<x.cols;++d) y[r][d]=x[r][d]*scale*(weight?scalar(weight,d):1.0f);
    }
    return y;
}
Matrix GemmaAudioEncoder::clipped_linear(const Matrix &x,const std::string &name) const {
    Matrix input=x;
    clip(input,scalar(loader.weight(name+".input_min"),0),scalar(loader.weight(name+".input_max"),0));
    auto y=ops.linear(input,name,false);
    clip(y,scalar(loader.weight(name+".output_min"),0),scalar(loader.weight(name+".output_max"),0));
    return y;
}
Matrix GemmaAudioEncoder::feed_forward(const Matrix &x,const std::string &p,const std::string &suffix) const {
    Matrix input=x; clip(input,-1e10f,1e10f);
    auto y=clipped_linear(rms(input,p+"ffn_norm"+suffix),p+"ffn_up"+suffix);
    silu(y); y=clipped_linear(y,p+"ffn_down"+suffix); clip(y,-1e10f,1e10f);
    y=rms(y,p+"ffn_post_norm"+suffix);
    for(size_t i=0;i<y.data.size();++i) y.data[i]=x.data[i]+0.5f*y.data[i];
    return y;
}
Matrix GemmaAudioEncoder::subsample(const Matrix &features) const {
    // Time-major [T,F,C]. Both convolutions use symmetric pad=1,stride=2.
    Matrix cur=features;
    int time=features.rows, freq=mel_bins, inc=1;
    for(int layer=0;layer<2;++layer) {
        const int nt=(time+1)/2,nf=(freq+1)/2,oc=channels[layer];
        Matrix patches(nt*nf,9*inc);
        for(int t=0;t<nt;++t) for(int f=0;f<nf;++f)
            for(int c=0;c<inc;++c) for(int kt=0;kt<3;++kt) for(int kf=0;kf<3;++kf) {
                const int it=2*t+kt-1,jf=2*f+kf-1;
                if(it>=0 && it<time && jf>=0 && jf<freq)
                    patches[t*nf+f][(c*3+kt)*3+kf]=cur[it][jf*inc+c];
            }
        const auto p="a.conv1d."+std::to_string(layer);
        auto convolved=convolve(patches,loader.weight(p+".weight"),oc,ops.threads());
        const auto *w=loader.weight(p+".norm.weight");
        Matrix next(nt,nf*oc);
        for(int r=0;r<nt*nf;++r) {
            double mean=0,var=0;
            for(int c=0;c<oc;++c) mean+=convolved[r][c];
            mean/=oc;
            for(int c=0;c<oc;++c) { const double diff=convolved[r][c]-mean; var+=diff*diff; }
            const float scale=static_cast<float>(1.0/std::sqrt(var/oc+epsilon));
            for(int c=0;c<oc;++c)
                next[r/nf][(r%nf)*oc+c]=std::max(0.0f,(convolved[r][c]-static_cast<float>(mean))*scale*scalar(w,c));
        }
        cur=std::move(next); time=nt;freq=nf;inc=oc;
    }
    return ops.linear(cur,"a.input_projection",false);
}
Matrix GemmaAudioEncoder::attention(const Matrix &x,const std::string &p,const Matrix &positions) const {
    const int dim=hidden/heads;
    auto q=clipped_linear(x,p+"attn_q"),k=clipped_linear(x,p+"attn_k"),v=clipped_linear(x,p+"attn_v");
    const auto relative=ops.linear(positions,p+"attn_k_rel",false);
    const float qs=static_cast<float>(1.0/std::sqrt(double(dim))/std::log(2.0));
    const float ks=static_cast<float>(std::log(1.0+std::exp(1.0))/std::log(2.0));
    const auto *scale=loader.weight(p+"per_dim_scale.weight");
    for(int t=0;t<x.rows;++t) for(int d=0;d<hidden;++d) {
        q[t][d]*=qs*scalar(scale,d%dim); k[t][d]*=ks;
    }
    Matrix y(x.rows,hidden);
    // Exact reduction of blocked [12,24] attention for right_context=0:
    // valid keys j are t-12<j<=t (the HF mask's left bound is strict).
    // Under the upstream relative shift,
    // (qi*24+s) / 25 == qi and (qi*24+s) % 25 == 12-(t-j).
    // Padded/invalid logits are -1e9 and have exactly zero softmax mass.
    for(int t=0;t<x.rows;++t) for(int h=0;h<heads;++h) {
        const int first=std::max(0,t-left_context+1),count=t-first+1;
        std::array<float,13> scores;
        float maximum=-std::numeric_limits<float>::infinity();
        for(int j=first;j<=t;++j) {
            float content=0,rpe=0;
            for(int d=0;d<dim;++d) {
                const int c=h*dim+d;
                content+=q[t][c]*k[j][c];
                rpe+=q[t][c]*relative[left_context-(t-j)][c];
            }
            scores[j-first]=50.0f*std::tanh((content+rpe)/50.0f);
            maximum=std::max(maximum,scores[j-first]);
        }
        float sum=0;
        for(int j=0;j<count;++j) { scores[j]=std::exp(scores[j]-maximum);sum+=scores[j]; }
        for(int j=first;j<=t;++j) for(int d=0;d<dim;++d)
            y[t][h*dim+d]+=scores[j-first]/sum*v[j][h*dim+d];
    }
    return clipped_linear(y,p+"attn_out");
}
Matrix GemmaAudioEncoder::encode_features(const Matrix &features) const {
    require(features.cols==mel_bins && features.rows>0 && features.rows<=3000,"invalid feature shape");
    finite(features);
    auto x=subsample(features);
    Matrix positions(13,hidden);
    const int half=hidden/2;
    for(int r=0;r<13;++r) for(int i=0;i<half;++i) {
        const float timescale=std::exp(static_cast<float>(-std::log(10000.0)*i/std::max(half-1,1)));
        const float phase=(12-r)*timescale;
        positions[r][i]=std::sin(phase);positions[r][i+half]=std::cos(phase);
    }
    for(int il=0;il<layers;++il) {
        const auto p="a.blk."+std::to_string(il)+".";
        x=feed_forward(x,p,"");
        auto y=x;clip(y,-1e10f,1e10f);
        y=attention(rms(y,p+"attn_pre_norm"),p,positions);clip(y,-1e10f,1e10f);
        add(x,rms(y,p+"attn_post_norm"));
        // GGUF conv_norm is HF pre_layer_norm; norm_conv is HF conv_norm.
        y=clipped_linear(rms(x,p+"conv_norm"),p+"conv_pw1");
        Matrix gated(y.rows,hidden);
        for(int t=0;t<y.rows;++t) for(int d=0;d<hidden;++d)
            gated[t][d]=y[t][d]/(1.0f+std::exp(-y[t][d+hidden]));
        Matrix conv(y.rows,hidden);
        const auto *w=loader.weight(p+"conv_dw.weight");
        for(int t=0;t<y.rows;++t) for(int d=0;d<hidden;++d)
            for(int tap=0;tap<5;++tap) if(t+tap>=4)
                conv[t][d]+=gated[t+tap-4][d]*scalar(w,static_cast<size_t>(d)*5+tap);
        clip(conv,-1e10f,1e10f);y=rms(conv,p+"norm_conv");silu(y);
        add(x,clipped_linear(y,p+"conv_pw2"));
        x=feed_forward(x,p,"_1");clip(x,-1e10f,1e10f);x=rms(x,p+"ln2");
    }
    x=ops.linear(x,"a.pre_encode.out",true);
    x=rms(x,loader.has_tensor_prefix("mm.a.soft_emb_norm.weight")?"mm.a.soft_emb_norm":"");
    x=ops.linear(x,"mm.a.input_projection",false);
    finite(x);
    return x;
}
Matrix GemmaAudioEncoder::encode(const std::vector<float> &pcm) const { return encode_features(log_mel(pcm)); }
} // namespace clap::detail
