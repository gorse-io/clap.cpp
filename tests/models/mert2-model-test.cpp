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
        ASSERT_TRUE(std::isfinite(actual[i])) << name + " contains a non-finite value";
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    std::cout << name << " max_abs_error=" << error << '\n';
    ASSERT_LE(error, tolerance) << name + " numerical mismatch";
}

static std::string model_path, fixture_path;

TEST(Mert2Model, OracleParity) {
    std::ifstream input(fixture_path);
    ASSERT_TRUE(input.is_open()) << "cannot open fixture: " << fixture_path;
    nlohmann::json fixture;
    input >> fixture;

    clap::Model model(model_path, 2);
    ASSERT_EQ(model.audio_sample_rate(), fixture.at("sample_rate").get<int>()) << "MERT2 sample rate";
    ASSERT_EQ(model.projection_dimension(), fixture.at("hidden_size").get<int>()) << "MERT2 hidden size";
    const float tolerance = fixture.at("tolerance");
    for (const auto &test : fixture.at("waveform_cases")) {
        const auto waveform = test.at("waveform").get<std::vector<float>>();
        const auto output = model.audio_frames(waveform, fixture.at("sample_rate"));
        ASSERT_EQ(output.frames, test.at("frames").get<int>()) << test.at("name").get<std::string>() + " frame count";
        ASSERT_EQ(output.dimension, fixture.at("hidden_size").get<int>()) << "MERT2 frame width";
        ASSERT_NO_FATAL_FAILURE(compare(output.values, test.at("expected_frames").get<std::vector<float>>(), tolerance,
                test.at("name")));
        ASSERT_NO_FATAL_FAILURE(compare(model.audio(waveform, fixture.at("sample_rate")),
                test.at("expected_pooled").get<std::vector<float>>(), tolerance,
                test.at("name").get<std::string>() + " pooled"));
        if (test.value("full_song", false)) {
            ASSERT_TRUE(waveform.size() > static_cast<size_t>(fixture.at("sample_rate").get<int>()) * 30) << "full-song fixture must exceed 30 seconds";
            ASSERT_TRUE(output.frames == waveform.size() / fixture.at("inputs_to_logits_ratio").get<size_t>()) << "full-song inference silently truncated the waveform";
        }
    }

    ASSERT_GT(fixture.at("minimum_input_samples").get<size_t>(), 0u);
    ASSERT_THROW(model.audio_frames(std::vector<float>(fixture.at("minimum_input_samples").get<size_t>() - 1),
                                   fixture.at("sample_rate")), std::exception)
        << "too-short MERT2 waveform must be rejected";
    auto invalid = fixture.at("waveform_cases").at(0).at("waveform").get<std::vector<float>>();
    ASSERT_FALSE(invalid.empty());
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_THROW(model.audio_frames(invalid, fixture.at("sample_rate")), std::exception)
        << "non-finite MERT2 waveform must be rejected";
    ASSERT_THROW(model.text("MERT2 has no text tower"), std::exception)
        << "MERT2 text input must be rejected";
    ASSERT_THROW(model.audio_features({0}, 1), std::exception)
        << "MERT2 precomputed feature input must be rejected";
    std::cout << "Native MERT-v2-FullSong parity passed\n";
}


int main(int argc, char **argv) {
    oracle_test::initialize(&argc, argv);
    if (argc != 3 && !GTEST_FLAG_GET(list_tests)) {
        std::cerr << "usage: mert2-model-test MODEL.gguf FIXTURE.json\n";
        return 1;
    }
    if (argc == 3) { model_path = argv[1]; fixture_path = argv[2]; }
    return RUN_ALL_TESTS();
}
