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
    if (error > tolerance) {
        for (size_t i = 0; i < actual.size(); ++i) if (std::abs(actual[i] - expected[i]) == error) {
            std::cerr << name << " worst index=" << i << " actual=" << actual[i] << " expected=" << expected[i] << '\n';
            break;
        }
    }
    ASSERT_LE(error, tolerance) << name + " numerical mismatch";
}
static std::string model_path, fixture_path;

TEST(ClapFrontend, SilenceLogMel) {
    const auto silent = clap::log_mel(std::vector<float>(480000, 0));
    ASSERT_EQ(silent.size(), 1001u * 64);
    ASSERT_NO_FATAL_FAILURE(compare(silent, std::vector<float>(1001 * 64, -100), 1e-5f, "silence log mel"));
}

TEST(ClapFrontend, RejectsInvalidWaveforms) {
    EXPECT_THROW(clap::log_mel({}), std::exception);
    EXPECT_THROW(clap::log_mel({0}, 16000), std::exception);
    EXPECT_THROW(clap::log_mel({std::numeric_limits<float>::quiet_NaN()}), std::exception);
}

TEST(ClapFrontend, ExplicitLongClipCrop) {
    std::vector<float> long_clip(480003, 0); long_clip[24000] = 0.5f;
    const std::vector<float> crop(long_clip.begin() + 2, long_clip.begin() + 480002);
    ASSERT_NO_FATAL_FAILURE(compare(clap::log_mel(long_clip, 48000, 2), clap::log_mel(crop), 0, "explicit long-clip crop"));
    EXPECT_THROW(clap::log_mel(long_clip, 48000, 4), std::exception);
}

TEST(ClapModel, RejectsMissingGGUF) {
    EXPECT_THROW(clap::Model("/this/path/does/not/exist.gguf"), std::exception);
}

TEST(ClapModel, OracleParity) {
    if (model_path.empty()) GTEST_SKIP() << "no model/fixture arguments supplied";
    std::ifstream input(fixture_path);
    ASSERT_TRUE(input.is_open()) << "cannot open fixture: " << fixture_path;
    nlohmann::json fixture; input >> fixture;
    clap::Model model(model_path, 2);
    const float embedding_tolerance = fixture.value("embedding_tolerance", 5e-5f);
    for (const auto &test : fixture.at("text_cases")) {
        const auto ids = model.tokenize(test.at("text").get<std::string>());
        ASSERT_EQ(ids, test.at("ids").get<std::vector<int>>()) << "tokenizer mismatch: " + test.at("text").get<std::string>();
        ASSERT_NO_FATAL_FAILURE(compare(model.text(test.at("padded_ids").get<std::vector<int>>()),
                test.at("expected").get<std::vector<float>>(), embedding_tolerance, "text embedding"));
    }
    ASSERT_NO_FATAL_FAILURE(compare(model.audio_features(fixture.at("features").get<std::vector<float>>(), fixture.at("frames")),
            fixture.at("expected_audio").get<std::vector<float>>(), embedding_tolerance, "audio embedding"));
    if (fixture.contains("waveform")) {
        ASSERT_NO_FATAL_FAILURE(compare(clap::log_mel(fixture.at("waveform").get<std::vector<float>>()),
                fixture.at("mel").get<std::vector<float>>(), 3e-4f, "waveform log mel"));
        if (fixture.value("end_to_end", false))
            ASSERT_NO_FATAL_FAILURE(compare(model.audio(fixture.at("waveform").get<std::vector<float>>()),
                    fixture.at("expected_audio").get<std::vector<float>>(), 5e-5f, "waveform embedding"));
    }
    std::cout << "Native CLAP parity passed\n";
}


int main(int argc, char **argv) {
    oracle_test::initialize(&argc, argv);
    if (argc != 1 && argc != 3) {
        std::cerr << "usage: clap-model-test [MODEL.gguf FIXTURE.json]\n";
        return 1;
    }
    if (argc == 3) { model_path = argv[1]; fixture_path = argv[2]; }
    return RUN_ALL_TESTS();
}
