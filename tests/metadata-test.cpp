#include "model-loader.h"
#include <gguf.h>
#include <nlohmann/json.hpp>
#include <cstdio>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    auto *ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", "test");
    gguf_set_val_u32(ctx, "test.count", 17);
    gguf_set_val_f32(ctx, "test.epsilon", 0.25f);
    gguf_set_val_bool(ctx, "test.flag", true);
    const char *tokens[] = {"<unk>", "▁hello"};
    gguf_set_arr_str(ctx, "test.tokens", tokens, 2);
    const float scores[] = {-1.5f, -2.5f};
    gguf_set_arr_data(ctx, "test.scores", GGUF_TYPE_FLOAT32, scores, 2);
    if (!gguf_write_to_file(ctx, argv[1], true)) return 1;
    gguf_free(ctx);
    try {
        clap::detail::ModelLoader loader;
        loader.load(argv[1]);
        if (!loader.has_metadata("test.count") || loader.has_metadata("test.missing")) return 2;
        if (loader.integer("test.count") != 17 || loader.number("test.epsilon") != 0.25 || !loader.boolean("test.flag")) return 3;
        if (loader.metadata_value("test.tokens") != nlohmann::json::array({"<unk>", "▁hello"})) return 4;
        if (loader.metadata_value("test.scores") != nlohmann::json::array({-1.5, -2.5})) return 5;
        for (const auto *key : {"test.epsilon", "test.flag", "test.missing"}) {
            bool rejected = false;
            try { (void)loader.integer(key); } catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) return 6;
        }
        bool rejected = false;
        try { (void)loader.boolean("test.count"); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) return 7;
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 8; }
    return 0;
}
