#include "clap.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}

static void compare(const std::vector<float> &actual, const std::vector<float> &expected,
                    float tolerance, const std::string &name) {
    require(actual.size() == expected.size(), name + " shape mismatch");
    float error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), name + " is not finite");
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    std::cout << name << " max_abs_error=" << error << '\n';
    require(error <= tolerance, name + " numerical mismatch");
}

int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: clamp3-model-test MODEL.gguf FIXTURE.json");
        std::ifstream input(argv[2]);
        nlohmann::json fixture; input >> fixture;
        clap::Model model(argv[1], 2);
        require(model.projection_dimension() == fixture.at("projection_dim"), "projection dimension");
        const float tolerance = fixture.value("embedding_tolerance", 8e-5f);
        for (const auto &test : fixture.at("tokenizer_cases")) {
            const auto ids = model.tokenize(test.at("text").get<std::string>(), test.value("max_length", 512));
            require(ids == test.at("ids").get<std::vector<int>>(), "tokenizer mismatch: " + test.at("text").get<std::string>());
        }
        for (const auto &test : fixture.at("text_cases")) {
            compare(model.text(test.at("ids").get<std::vector<int>>()),
                    test.at("expected").get<std::vector<float>>(), tolerance, test.at("name"));
            if (test.contains("utf8"))
                compare(model.text(test.at("utf8").get<std::string>()),
                        test.at("expected").get<std::vector<float>>(), tolerance, test.at("name"));
        }
        for (const auto &test : fixture.at("audio_cases"))
            compare(model.audio_features(test.at("features").get<std::vector<float>>(), test.at("frames")),
                    test.at("expected").get<std::vector<float>>(), tolerance, test.at("name"));

        bool rejected = false;
        try { model.audio(std::vector<float>(48000, 0), 48000); }
        catch (const std::exception &e) {
            rejected = std::string(e.what()).find("MERT-v1-95M") != std::string::npos;
        }
        require(rejected, "CLaMP3 raw waveform must clearly direct users to MERT-v1-95M features");
        rejected = false;
        try { model.audio_features({0}, 1); } catch (const std::exception &) { rejected = true; }
        require(rejected, "incorrect MERT feature shape must be rejected");
        rejected = false;
        auto nonfinite = fixture.at("audio_cases").at(0).at("features").get<std::vector<float>>();
        nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
        try { model.audio_features(nonfinite, fixture.at("audio_cases").at(0).at("frames")); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "non-finite MERT features must be rejected");
        std::cout << "Native CLaMP3 parity passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
