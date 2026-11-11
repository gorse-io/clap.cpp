// Optional independent reference: compile against an external pinned llama.cpp.
// Never linked into clap or required by default CTest.
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <llama.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Test-only arithmetic adapter, not a converter: retain standard GGUF weights
// and decode just the current block-matmul operand, matching CLAP_ACTIVATION_F32.
// The upstream llama graph and backend remain independent of clap.
struct F32Operands {
    ggml_context *context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *node = nullptr;
    ggml_tensor *original = nullptr;

    void clear() {
        if (node)
            node->src[0] = original;
        if (buffer)
            ggml_backend_buffer_free(buffer);
        if (context)
            ggml_free(context);
        context = nullptr;
        buffer = nullptr;
        node = nullptr;
    }
    ~F32Operands() {
        node = nullptr; // The llama context/graph is destroyed first on an exception.
        clear();
    }

    static bool evaluate(ggml_tensor *tensor, bool ask, void *data) {
        auto &self = *static_cast<F32Operands *>(data);
        if (!ask) {
            self.clear();
            return true;
        }
        if (tensor->op != GGML_OP_MUL_MAT || !tensor->src[0] || tensor->src[0]->type == GGML_TYPE_F32 ||
            std::string(ggml_get_name(tensor->src[0])).find("blk.") != 0)
            return false;
        auto *weight = tensor->src[0];
        const auto *traits = ggml_get_type_traits(weight->type);
        if (!traits || !traits->to_float || !ggml_is_contiguous(weight))
            throw std::runtime_error("unsupported F32 reference operand");
        self.context = ggml_init({ggml_tensor_overhead(), nullptr, true});
        if (!self.context)
            throw std::runtime_error("F32 reference metadata allocation failed");
        auto *decoded =
            ggml_new_tensor_4d(self.context, GGML_TYPE_F32, weight->ne[0], weight->ne[1], weight->ne[2], weight->ne[3]);
        self.buffer = ggml_backend_alloc_ctx_tensors_from_buft(self.context, ggml_backend_cpu_buffer_type());
        if (!self.buffer)
            throw std::runtime_error("F32 reference operand allocation failed");
        traits->to_float(weight->data, static_cast<float *>(decoded->data), ggml_nelements(weight));
        self.original = weight;
        self.node = tensor;
        tensor->src[0] = decoded;
        return true;
    }
};

int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--f32-operands"))
            throw std::runtime_error("usage: qwen3-llama-reference MODEL.gguf TEXTS.json [--f32-operands]");
        nlohmann::json texts;
        std::ifstream(argv[2]) >> texts;
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        // Match clap's standard mmap-backed CPU weights, not CPU_REPACK kernels.
        mp.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp),
                                                                        llama_model_free);
        if (!model)
            throw std::runtime_error("llama.cpp model loading failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_threads = 2;
        cp.n_threads_batch = 2;
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_LAST;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        // clap's full-sequence attention retains K/V in F32 (no packed KV cache).
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        F32Operands operands;
        if (argc == 4) {
            cp.cb_eval = F32Operands::evaluate;
            cp.cb_eval_user_data = &operands;
        }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx)
            throw std::runtime_error("llama.cpp context creation failed");
        const auto *vocab = llama_model_get_vocab(model.get());
        const int width = llama_model_n_embd_out(model.get());
        auto cases = nlohmann::json::array();
        for (const auto &entry : texts) {
            const std::string text = entry;
            std::vector<llama_token> ids(text.size() + 32);
            const int count =
                llama_tokenize(vocab, text.data(), int(text.size()), ids.data(), int(ids.size()), true, true);
            if (count <= 0 || count > 512)
                throw std::runtime_error("invalid reference token count");
            ids.resize(count);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            auto batch = llama_batch_init(count, 0, 1);
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.token[i] = ids[i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = true;
            }
            const int status = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            if (status != 0)
                throw std::runtime_error("llama.cpp decode failed");
            const float *embedding = llama_get_embeddings_seq(ctx.get(), 0);
            if (!embedding)
                throw std::runtime_error("llama.cpp returned no sequence embedding");
            double squared = 0;
            for (int i = 0; i < width; ++i)
                squared += double(embedding[i]) * embedding[i];
            std::vector<float> normalized(width);
            for (int i = 0; i < width; ++i)
                normalized[i] = float(embedding[i] / std::sqrt(squared));
            cases.push_back({{"text", text}, {"tokens", ids}, {"expected", normalized}});
        }
        std::cout << cases.dump() << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
