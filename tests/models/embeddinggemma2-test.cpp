#include "clap.h"
#include "../c-api-helpers.h"
using namespace c_api_test;
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
int main(int argc, char **argv) {
    try {
        require(argc >= 3, "usage: embeddinggemma2-test MODEL FIXTURE [reject-message]");
        if (argc == 4) {
            try {
                CLAP *raw = nullptr;
                check(clap_create(argv[1], nullptr, &raw));
                Model model(raw, clap_destroy);
            } catch (const std::exception &e) {
                require(std::string(e.what()).find(argv[3]) != std::string::npos, e.what());
                return 0;
            }
            throw std::runtime_error("invalid model accepted");
        }
        auto options = clap_default_options();
        options.threads = 2;
        CLAP *raw = nullptr;
        check(clap_create(argv[1], &options, &raw));
        Model model(raw, clap_destroy);
        CLAP *missing = nullptr;
        const auto missing_path = std::string(argv[1]) + ".missing-mmproj";
        require(clap_create_with_mmproj(argv[1], missing_path.c_str(), &options, &missing) != CLAP_OK,
                "missing mmproj accepted");
        require(missing == nullptr && std::string(clap_last_error()).find("missing-mmproj") != std::string::npos,
                "missing mmproj path was not reported");
        std::ifstream input(argv[2]);
        nlohmann::json fixture;
        input >> fixture;
        float maximum = 0;
        for (const auto &test : fixture.at("text_cases")) {
            const auto text = test.at("text").get<std::string>();
            require(take_buffer<int>([&](int **out, size_t *count) {
                        return clap_tokenize(model.get(), text.c_str(), fixture.at("context").get<size_t>(), out,
                                             count);
                    }) == test.at("ids").get<std::vector<int>>(),
                    "tiny Gemma4 tokenizer mismatch");
            const auto actual = take_buffer<float>(
                [&](float **out, size_t *count) { return clap_encode_text(model.get(), text.c_str(), out, count); });
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
            const auto actual = take_buffer<float>([&](float **out, size_t *count) {
                return clap_encode_tokens(model.get(), ids.data(), ids.size(), out, count);
            });
            require(actual.size() == expected.size(), "projection dimension mismatch");
            for (size_t i = 0; i < actual.size(); ++i) {
                require(std::isfinite(actual[i]), "non-finite embedding");
                maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
            }
            require(actual == take_buffer<float>([&](float **out, size_t *count) {
                        return clap_encode_tokens(model.get(), ids.data(), ids.size(), out, count);
                    }),
                    "repeated inference changed output");
        }
        std::cout << "EmbeddingGemma2 independent scalar oracle max_abs_error=" << maximum << '\n';
        require(maximum <= fixture.value("tolerance", 8e-5f), "independent numerical oracle mismatch");
        for (const auto &ids : {std::vector<int>{}, std::vector<int>{-1}, std::vector<int>{0, 0, 0},
                                std::vector<int>{fixture.at("vocab").get<int>()},
                                std::vector<int>(fixture.at("context").get<int>() + 1, 3)}) {
            bool rejected = false;
            try {
                (void)take_buffer<float>([&](float **out, size_t *count) {
                    return clap_encode_tokens(model.get(), ids.data(), ids.size(), out, count);
                });
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected, "invalid token input accepted");
        }
        bool rejected = false;
        try {
            (void)take_buffer<float>([&](float **out, size_t *count) {
                const std::vector<float> pcm(48000, 0.1f);
                return clap_encode_audio(model.get(), pcm.data(), pcm.size(), 48000, 0, out, count);
            });
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "text-only model accepted audio");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
