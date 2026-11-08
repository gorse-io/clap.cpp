#include "vocab.h"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        const nlohmann::json tokenizer = {{"model",
                                           {{"type", "BPE"},
                                            {"vocab", {{"<", 0}, {"|", 1}, {"e", 2}, {"o", 3}, {"s", 4}, {">", 5}}},
                                            {"merges", nlohmann::json::array()}}},
                                          {"added_tokens", {{{"id", 6}, {"content", "<|eos|>"}, {"special", true}}}}};
        const clap::detail::ByteBpeVocabulary legacy(tokenizer, 7, 8, 20);
        const clap::detail::ByteBpeVocabulary ordinary(tokenizer, 7, 8, 20, true);
        if (legacy.tokenize("<|eos|>", 20) != std::vector<int>{7, 6, 8})
            throw std::runtime_error("default special-token recognition changed");
        if (ordinary.tokenize("<|eos|>", 20) != std::vector<int>{7, 0, 1, 2, 3, 4, 1, 5, 8})
            throw std::runtime_error("CLAP2 special-token markup must be ordinary text");
        if (ordinary.tokenize("<|eos|>", 4) != std::vector<int>{7, 0, 1, 8})
            throw std::runtime_error("ordinary-text truncation must preserve EOS");
        if (ordinary.tokenize("", 20) != std::vector<int>{7, 8})
            throw std::runtime_error("empty text must contain BOS/EOS");
        std::cout << "byte BPE ordinary-text/special-token regression passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
