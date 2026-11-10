#include "vocab.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected tokenizer fixture");
        nlohmann::json f;
        std::ifstream(argv[1]) >> f;
        // Negative BOS means Qwen's actual no-BOS/EOS postprocessor.
        clap::detail::ByteBpeVocabulary vocab(f.at("tokenizer"), -1, 151643, 32768);
        for (const auto &c : f.at("cases")) {
            const auto actual = vocab.tokenize(c.at("text"), 32768);
            const auto expected = c.at("tokens").get<std::vector<int>>();
            if (actual != expected) {
                std::cerr << c.at("text") << "\nactual " << nlohmann::json(actual)
                          << "\nexpected " << nlohmann::json(expected) << '\n';
                throw std::runtime_error("published Qwen3 tokenizer mismatch");
            }
        }
        if (vocab.tokenize("hello", 1) != std::vector<int>{151643})
            throw std::runtime_error("one-token truncation must retain EOS");
        std::cout << "Published Qwen3 tokenizer Unicode/merges/specials passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
