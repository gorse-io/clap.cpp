#include "clap.h"
#include "../c-api-helpers.h"

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
        require(argc == 3, "usage: clap2-model-test MODEL.gguf FIXTURE.json");
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
        require(clap_audio_sample_rate(model.get()) == fixture.at("sample_rate"), "CLAP2 sample rate");
        require(clap_projection_dimension(model.get()) == fixture.at("projection_dim"), "CLAP2 projection width");
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
                throw std::runtime_error("CLAP2 tokenizer mismatch");
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
        for (const auto &test : fixture.at("waveform_cases")) {
            const auto path =
                std::filesystem::path(argv[2]).parent_path() / test.at("waveform_file").get<std::string>();
            const size_t samples = test.at("samples");
            std::vector<float> wave(samples);
            std::ifstream pcm(path, std::ios::binary);
            require(bool(pcm), "cannot open waveform " + path.string());
            pcm.read(reinterpret_cast<char *>(wave.data()), static_cast<std::streamsize>(samples * sizeof(float)));
            require(pcm.gcount() == static_cast<std::streamsize>(samples * sizeof(float)), "truncated PCM fixture");
            // Fixture files explicitly use little-endian IEEE F32.
            const unsigned short endian = 1;
            if (*reinterpret_cast<const unsigned char *>(&endian) != 1) {
                auto *bytes = reinterpret_cast<unsigned char *>(wave.data());
                for (size_t i = 0; i < samples; ++i)
                    std::reverse(bytes + 4 * i, bytes + 4 * i + 4);
            }
            if (test.at("name") == "long-no-autocrop")
                require(samples > size_t(fixture.at("sample_rate").get<int>()) * 30, "long fixture <=30 seconds");
            compare(take_buffer<float>([&](float **out, size_t *count) {
                        return clap_encode_audio(model.get(), wave.data(), wave.size(), fixture.at("sample_rate"), 0,
                                                 out, count);
                    }),
                    test.at("expected").get<std::vector<float>>(), test.value("tolerance", tolerance), test.at("name"));
            float *out = nullptr;
            size_t count = 0;
            const auto status =
                clap_encode_audio(model.get(), wave.data(), wave.size(), fixture.at("sample_rate"), 1, &out, &count);
            clap_free(out);
            require(status != CLAP_OK, "CLAP2 must reject legacy CLAP crop offsets");
        }
        std::cout << "Native CLAP2 official-oracle parity passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
