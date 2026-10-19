#include "vocab.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: clap-unigram-test TOKENIZER.json CASES.json");
        nlohmann::json tokenizer, cases;
        std::ifstream input(argv[1]); input >> tokenizer;
        std::ifstream reference(argv[2]); reference >> cases;
        clap::detail::UnigramVocabulary vocab(tokenizer, 0, 2, 3, 4096);
        for (const auto &test : cases) {
            auto actual = vocab.tokenize(test.at("text").get<std::string>(), test.value("max_length", 4096));
            auto expected = test.at("ids").get<std::vector<int>>();
            if (actual != expected) {
                std::cerr << "tokenizer mismatch: " << test.at("text") << "\nactual: " << nlohmann::json(actual)
                          << "\nexpected: " << nlohmann::json(expected) << '\n';
                return 1;
            }
        }
        auto reject = [&](const nlohmann::json &invalid) {
            bool rejected = false;
            try { clap::detail::UnigramVocabulary unsupported(invalid, 0, 2, 3, 4096); }
            catch (const std::exception &) { rejected = true; }
            if (!rejected) throw std::runtime_error("unsupported tokenizer metadata was accepted");
        };
        auto invalid = tokenizer;
        invalid["pre_tokenizer"] = nullptr; reject(invalid);
        invalid = tokenizer;
        invalid["pre_tokenizer"]["pretokenizers"][1]["prepend_scheme"] = "never"; reject(invalid);
        invalid = tokenizer;
        invalid["post_processor"] = nullptr; reject(invalid);
        invalid = tokenizer;
        invalid["post_processor"]["single"] = nlohmann::json::array(); reject(invalid);
        invalid = tokenizer;
        invalid["model"]["unk_id"] = 4; reject(invalid);
        invalid = tokenizer;
        invalid["model"]["byte_fallback"] = true; reject(invalid);
        std::cout << "Unigram tokenizer parity passed: " << cases.size() << " cases; invalid metadata rejected\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
