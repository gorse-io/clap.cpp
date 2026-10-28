#include "clap.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main(int argc, char **argv) {
    try {
        require(argc >= 3, "usage: embeddinggemma2-test MODEL FIXTURE [reject-message]");
        if (argc == 4) {
            try { clap::CLAP model(argv[1], 2); }
            catch (const std::exception &e) {
                require(std::string(e.what()).find(argv[3]) != std::string::npos, e.what());
                return 0;
            }
            throw std::runtime_error("invalid model accepted");
        }
        clap::CLAP model(argv[1], 2);
        std::ifstream input(argv[2]);
        nlohmann::json fixture; input >> fixture;
        float maximum = 0;
        for (const auto &test : fixture.at("text_cases")) {
            const auto text = test.at("text").get<std::string>();
            require(model.tokenize(text, fixture.at("context").get<size_t>()) == test.at("ids").get<std::vector<int>>(), "tiny Gemma4 tokenizer mismatch");
            const auto actual = model.encode_text(text);
            const auto expected = test.at("embedding").get<std::vector<float>>();
            require(actual.size() == expected.size(), "text projection dimension mismatch");
            for (size_t i = 0; i < actual.size(); ++i) {
                require(std::isfinite(actual[i]), "non-finite text embedding");
                maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
            }
        }
        for (const auto &test : fixture.at("cases")) {
            const auto ids = test.at("ids").get<std::vector<int>>();
            const auto expected = test.at("embedding").get<std::vector<float>>();
            const auto actual = model.encode_text(ids);
            require(actual.size() == expected.size(), "projection dimension mismatch");
            for (size_t i = 0; i < actual.size(); ++i) {
                require(std::isfinite(actual[i]), "non-finite embedding");
                maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
            }
            require(actual == model.encode_text(ids), "repeated inference changed output");
        }
        std::cout << "EmbeddingGemma2 independent scalar oracle max_abs_error=" << maximum << '\n';
        require(maximum <= fixture.value("tolerance", 8e-5f), "independent numerical oracle mismatch");
        for (const auto &ids : {std::vector<int>{}, std::vector<int>{-1}, std::vector<int>{0, 0, 0},
                std::vector<int>{fixture.at("vocab").get<int>()},
                std::vector<int>(fixture.at("context").get<int>() + 1, 3)}) {
            bool rejected = false;
            try { (void) model.encode_text(ids); } catch (const std::exception &) { rejected = true; }
            require(rejected, "invalid token input accepted");
        }
        bool rejected = false;
        try { (void) model.encode_audio(std::vector<float>(48000, 0.1f)); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "text-only model accepted audio");
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
