#include "../c-api-helpers.h"
#include "clap.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace c_api_test;

static void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

static void compare(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance,
                    const std::string &name) {
    require(actual.size() == expected.size(), name + " shape mismatch");
    float error = 0;
    double squared_norm = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), name + " contains non-finite data");
        error = std::max(error, std::abs(actual[i] - expected[i]));
        squared_norm += double(actual[i]) * actual[i];
    }
    std::cout << name << " max_abs_error=" << error << '\n';
    require(error <= tolerance, name + " numerical mismatch");
    require(std::abs(squared_norm - 1.0) < 2e-5, name + " must be unit normalized");
}

int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: test-qwen3-model MODEL.gguf FIXTURE.json");
        std::ifstream input(argv[2]);
        require(bool(input), "cannot open fixture");
        nlohmann::json fixture;
        input >> fixture;
        clap_runtime_options options = clap_default_options();
        options.threads = 2;
        options.device = "cpu";
        options.activation_precision = CLAP_ACTIVATION_F32;
        CLAP *raw = nullptr;
        check(clap_create(argv[1], &options, &raw));
        Model model(raw, clap_destroy);
        require(clap_audio_sample_rate(model.get()) == fixture.at("sample_rate"), "Qwen3 sample rate");
        require(clap_projection_dimension(model.get()) == fixture.at("projection_dim"), "Qwen3 projection width");
        const float tolerance = fixture.at("tolerance");
        const size_t context = fixture.at("context_length");
        for (const auto &test : fixture.at("text_cases")) {
            const std::string text = test.at("text");
            auto ids = take_buffer<int>([&](int **out, size_t *count) {
                return clap_tokenize(model.get(), text.c_str(), context, out, count);
            });
            const auto expected_ids = test.at("tokens").get<std::vector<int>>();
            if (ids != expected_ids) {
                std::cerr << "text=" << nlohmann::json(text) << "\nactual=" << nlohmann::json(ids)
                          << "\nexpected=" << nlohmann::json(expected_ids) << '\n';
                throw std::runtime_error("Qwen3 tokenizer mismatch");
            }
            const auto expected = test.at("expected").get<std::vector<float>>();
            compare(take_buffer<float>([&](float **out, size_t *count) {
                        return clap_encode_text(model.get(), text.c_str(), out, count);
                    }),
                    expected, tolerance, "text " + text);
            compare(take_buffer<float>([&](float **out, size_t *count) {
                        return clap_encode_tokens(model.get(), ids.data(), ids.size(), out, count);
                    }),
                    expected, tolerance, "explicit text tokens");
        }
        for (const auto &test : fixture.at("token_cases")) {
            const auto ids = test.at("tokens").get<std::vector<int>>();
            compare(take_buffer<float>([&](float **out, size_t *count) {
                        return clap_encode_tokens(model.get(), ids.data(), ids.size(), out, count);
                    }),
                    test.at("expected").get<std::vector<float>>(), tolerance, test.at("name"));
        }
        float pcm[2] = {0.1f, -0.1f};
        float *out = nullptr;
        size_t count = 0;
        require(clap_encode_audio(model.get(), pcm, 2, 16000, 0, &out, &count) == CLAP_ERROR && !out && !count,
                "Qwen3 must reject audio and clear outputs");
        clap_audio_frames frames{};
        require(clap_get_audio_frames(model.get(), pcm, 2, 16000, &frames) == CLAP_ERROR && !frames.values,
                "Qwen3 must reject frames");
        for (const auto &ids : std::vector<std::vector<int>>{{-1}, {266}, {257, 257}, std::vector<int>(129, 3)}) {
            require(clap_encode_tokens(model.get(), ids.data(), ids.size(), &out, &count) != CLAP_OK && !out && !count,
                    "Qwen3 must reject invalid tokens");
        }
        const auto short_ids = take_buffer<int>(
            [&](int **out, size_t *count) { return clap_tokenize(model.get(), "ab12", 2, out, count); });
        require(short_ids == std::vector<int>({265, 256}), "Qwen3 truncation must retain EOS");
        std::cout << "Native Qwen3 scalar oracle and public API passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
