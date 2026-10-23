#include "model-loader.h"
#include "model-ops.h"
#include "quantize.h"
#include <ggml.h>
#include <gguf.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace clap::detail;
static void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using File = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
struct Role { const char *name; bool high; };
static std::vector<Role> roles(const std::string &arch) {
    if (arch == "mert") return {{"projection.projection.weight",true}, {"layers.0.attention.q_proj.weight",false}, {"layers.0.attention.v_proj.weight",true}, {"layers.0.ffn.out.weight",true}};
    if (arch == "mert2") return {{"subsampling_module.0.convnext_layers.0.pointwise_block.1.weight",false}, {"subsampling_module.0.convnext_layers.0.pointwise_block.4.weight",true}, {"layers.0.attn.query_proj.weight",false}, {"layers.0.attn.value_proj.weight",true}, {"layers.0.ffn1.w_2.weight",true}, {"layers.0.ffn2.w_2.weight",true}};
    auto result = std::vector<Role>{{"conv.linear.weight",true}, {"conformer.layers.0.self_attn.linear_q.weight",false}, {"conformer.layers.0.self_attn.linear_v.weight",true}, {"conformer.layers.0.ffn1.output_dense.weight",true}, {"conformer.layers.0.ffn2.output_dense.weight",true}};
    if (arch == "muq_mulan") for (auto role : std::vector<Role>{{"text.embeddings.word_embeddings.weight",false}, {"text.encoder.layer.0.attention.self.value.weight",true}, {"text.encoder.layer.0.intermediate.dense.weight",false}, {"text.encoder.layer.0.output.dense.weight",true}, {"adapter.layers.0.0.to_q.weight",false}, {"adapter.layers.0.0.to_kv.weight",true}, {"adapter.layers.0.0.to_out.0.weight",false}, {"adapter.layers.0.1.1.weight",false}, {"adapter.layers.0.1.4.weight",true}, {"audio_proj.weight",true}, {"audio_latent.weight",true}, {"text_proj.weight",true}}) result.push_back(role);
    return result;
}
static void fixture(const std::string &path, const std::string &arch, ggml_type type, const std::vector<Role> &dense, bool bad = false) {
    Context ctx(ggml_init({16*1024*1024,nullptr,false}),ggml_free);
    File file(gguf_init_empty(),gguf_free);
    gguf_set_val_str(file.get(),"general.architecture",arch.c_str());
    auto add = [&](const char *name, std::vector<int64_t> shape, ggml_type storage) {
        auto *t=ggml_new_tensor(ctx.get(),storage,static_cast<int>(shape.size()),shape.data());
        ggml_set_name(t,name); std::vector<float> values(ggml_nelements(t));
        for (size_t i=0;i<values.size();++i) values[i]=0.1f*std::sin(float(i)*0.13f)+0.02f;
        if (storage==GGML_TYPE_F32) std::memcpy(t->data,values.data(),ggml_nbytes(t));
        else if (storage==GGML_TYPE_F16) ggml_fp32_to_fp16_row(values.data(),static_cast<ggml_fp16_t *>(t->data),values.size());
        else ggml_get_type_traits(storage)->from_float_ref(values.data(),t->data,values.size());
        gguf_add_tensor(file.get(),t);
    };
    for (const auto &role:dense) add(role.name,{256,4},type);
    add("frontend.filters",{256,2},GGML_TYPE_F32);
    add("feature_extractor.mel_scale.fb",{256,2},GGML_TYPE_F32);
    add("norm.weight",{256},GGML_TYPE_F32);
    add("conv.kernel.weight",{256,2,3},bad?GGML_TYPE_Q8_0:type);
    add("unaligned.weight",{17,3},type);
    require(gguf_write_to_file(file.get(),path.c_str(),false),"fixture write");
}
int main(int argc,char **argv) {
    try {
        require(argc==2,"usage: all-quantization-test DIRECTORY"); std::filesystem::create_directories(argv[1]);
        for (const std::string arch:{"mert","mert2","muq","muq_mulan"}) for (auto type:{GGML_TYPE_F32,GGML_TYPE_F16}) {
            const auto dense=roles(arch); const std::string source=std::string(argv[1])+"/"+arch+std::to_string(type)+".gguf";
            fixture(source,arch,type,dense);
            for (const std::string preset:{"Q8_0","Q4_K_M"}) {
                const auto output=source+"."+preset; std::filesystem::remove(output);
                require(run_quantize(source,output,preset)==0,arch+" quantization refused");
                ModelLoader loader; loader.load(output);
                bool uncovered=false;
                try { loader.validate_tensor_coverage(); } catch (const std::exception &) { uncovered=true; }
                require(uncovered,arch+" skipped strict tensor shape coverage");

                // Read raw source tensor bytes independently of inference promotion.
                ggml_context *raw=nullptr; File original(gguf_init_from_file(source.c_str(),{false,&raw}),gguf_free); Context weights(raw,ggml_free);
                int q4=0,q6=0;
                for (const auto &role:dense) {
                    auto *a=ggml_get_tensor(raw,role.name), *b=loader.weight(role.name);
                    auto expected_type=preset=="Q8_0"?GGML_TYPE_Q8_0:(role.high?GGML_TYPE_Q6_K:GGML_TYPE_Q4_K);
                    require(b->type==expected_type,arch+" incorrect role policy: "+role.name);
                    q4+=b->type==GGML_TYPE_Q4_K; q6+=b->type==GGML_TYPE_Q6_K;
                    std::vector<float> values(ggml_nelements(a)),decoded(values.size());
                    for(size_t i=0;i<values.size();++i) values[i]=scalar(a,i);
                    std::vector<uint8_t> expected(ggml_nbytes(b)); ggml_get_type_traits(b->type)->from_float_ref(values.data(),expected.data(),values.size());
                    require(std::memcmp(b->data,expected.data(),expected.size())==0,arch+" noncanonical bytes");
                    ggml_get_type_traits(b->type)->to_float(b->data,decoded.data(),decoded.size());
                    for(size_t i=0;i<decoded.size();++i) require(scalar(b,i)==decoded[i],"scalar decode");
                    TensorOps ops(loader,2); Matrix x(3,256);
                    for(size_t i=0;i<x.data.size();++i) x.data[i]=std::sin(float(i)*0.071f)*0.17f;
                    auto y=ops.linear(x,std::string(role.name).substr(0,std::strlen(role.name)-7),false);
                    for(int r=0;r<x.rows;++r) for(int c=0;c<y.cols;++c) {
                        float expected_y=0;for(int j=0;j<x.cols;++j)expected_y+=x[r][j]*decoded[c*x.cols+j];
                        require(std::abs(y[r][c]-expected_y)<2e-5f,arch+" storage-rounded F32 activation matmul drift");
                    }
                    require(loader.weight(role.name)->type==expected_type,"inference materialized mapped quantized weight");
                }
                require(preset!="Q4_K_M" || (q4>0&&q6>0),"must exercise both Q4/Q6");
                for(const auto *name:{"frontend.filters","feature_extractor.mel_scale.fb","norm.weight","conv.kernel.weight","unaligned.weight"}) {
                    // Raw output must preserve the original type/bytes, including F16 MuQ.
                    ggml_context *qraw=nullptr;File qfile(gguf_init_from_file(output.c_str(),{false,&qraw}),gguf_free);Context qweights(qraw,ggml_free);
                    auto *a=ggml_get_tensor(raw,name),*b=ggml_get_tensor(qraw,name);
                    require(a->type==b->type&&ggml_nbytes(a)==ggml_nbytes(b)&&std::memcmp(a->data,b->data,ggml_nbytes(a))==0,arch+" noneligible storage changed: "+name);
                }
                std::cout<<arch<<" "<<type<<" "<<preset<<" canonical bytes, roles, float preservation and native F32 activation matmul passed\n";
            }
            const auto bad=source+".bad"; fixture(bad,arch,type,dense,true);
            bool rejected=false;try{ModelLoader loader;loader.load(bad);}catch(const std::exception &){rejected=true;}
            require(rejected,"quantized convolution accepted: "+arch);
        }
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
