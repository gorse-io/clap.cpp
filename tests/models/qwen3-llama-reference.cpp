// Optional independent reference: compile against an external pinned llama.cpp.
// Never linked into clap or required by default CTest.
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <llama.h>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::runtime_error("usage: qwen3-llama-reference MODEL.gguf TEXTS.json");
        nlohmann::json texts;
        std::ifstream(argv[2]) >> texts;
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
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
