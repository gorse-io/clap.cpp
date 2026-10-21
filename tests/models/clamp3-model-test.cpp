#include "clap.h"
#include "../oracle-gtest.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void compare(const std::vector<float> &actual, const std::vector<float> &expected,
                    float tolerance, const std::string &name) {
    ASSERT_EQ(actual.size(), expected.size()) << name + " shape mismatch";
    float error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i])) << name + " is not finite";
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    std::cout << name << " max_abs_error=" << error << '\n';
    ASSERT_LE(error, tolerance) << name + " numerical mismatch";
}

static std::string model_path, fixture_path;

TEST(Clamp3Model, OracleParity) {
    std::ifstream input(fixture_path);
    ASSERT_TRUE(input.is_open()) << "cannot open fixture: " << fixture_path;
    nlohmann::json fixture; input >> fixture;
    clap::Model model(model_path, 2);
    ASSERT_EQ(model.projection_dimension(), fixture.at("projection_dim").get<int>()) << "projection dimension";
    const float tolerance = fixture.value("embedding_tolerance", 8e-5f);
    for (const auto &test : fixture.at("tokenizer_cases")) {
        const auto ids = model.tokenize(test.at("text").get<std::string>(), test.value("max_length", 512));
        ASSERT_EQ(ids, test.at("ids").get<std::vector<int>>()) << "tokenizer mismatch: " + test.at("text").get<std::string>();
    }
    for (const auto &test : fixture.at("text_cases")) {
        ASSERT_NO_FATAL_FAILURE(compare(model.text(test.at("ids").get<std::vector<int>>()),
                test.at("expected").get<std::vector<float>>(), tolerance, test.at("name")));
        if (test.contains("utf8"))
            ASSERT_NO_FATAL_FAILURE(compare(model.text(test.at("utf8").get<std::string>()),
                    test.at("expected").get<std::vector<float>>(), tolerance, test.at("name")));
    }
    for (const auto &test : fixture.at("audio_cases"))
        ASSERT_NO_FATAL_FAILURE(compare(model.audio_features(test.at("features").get<std::vector<float>>(), test.at("frames")),
                test.at("expected").get<std::vector<float>>(), tolerance, test.at("name")));

    bool rejected = false;
    try { model.audio(std::vector<float>(48000, 0), 48000); }
    catch (const std::exception &e) {
        rejected = std::string(e.what()).find("MERT-v1-95M") != std::string::npos;
    }
    ASSERT_TRUE(rejected) << "CLaMP3 raw waveform must clearly direct users to MERT-v1-95M features";
    ASSERT_THROW(model.audio_features({0}, 1), std::exception)
        << "incorrect MERT feature shape must be rejected";
    auto nonfinite = fixture.at("audio_cases").at(0).at("features").get<std::vector<float>>();
    ASSERT_FALSE(nonfinite.empty());
    nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_THROW(model.audio_features(nonfinite, fixture.at("audio_cases").at(0).at("frames")), std::exception)
        << "non-finite MERT features must be rejected";
    std::cout << "Native CLaMP3 parity passed\n";
}


int main(int argc, char **argv) {
    oracle_test::initialize(&argc, argv);
    if (argc != 3 && !GTEST_FLAG_GET(list_tests)) {
        std::cerr << "usage: clamp3-model-test MODEL.gguf FIXTURE.json\n";
        return 1;
    }
    if (argc == 3) { model_path = argv[1]; fixture_path = argv[2]; }
    return RUN_ALL_TESTS();
}
