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
        require(std::isfinite(actual[i]), name + " contains a non-finite value");
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    std::cout << name << " max_abs_error=" << error << '\n';
    require(error <= tolerance, name + " numerical mismatch");
}

int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: mert2-model-test MODEL.gguf FIXTURE.json");
        std::ifstream input(argv[2]);
        nlohmann::json fixture;
        input >> fixture;

        clap::Model model(argv[1], 2);
        require(model.audio_sample_rate() == fixture.at("sample_rate"), "MERT2 sample rate");
        require(model.projection_dimension() == fixture.at("hidden_size"), "MERT2 hidden size");
        const float tolerance = fixture.at("tolerance");
        for (const auto &test : fixture.at("waveform_cases")) {
            const auto waveform = test.at("waveform").get<std::vector<float>>();
            const auto output = model.audio_frames(waveform, fixture.at("sample_rate"));
            require(output.frames == test.at("frames"), test.at("name").get<std::string>() + " frame count");
            require(output.dimension == fixture.at("hidden_size"), "MERT2 frame width");
            compare(output.values, test.at("expected_frames").get<std::vector<float>>(), tolerance,
                    test.at("name"));
            compare(model.audio(waveform, fixture.at("sample_rate")),
                    test.at("expected_pooled").get<std::vector<float>>(), tolerance,
                    test.at("name").get<std::string>() + " pooled");
            if (test.value("full_song", false)) {
                require(waveform.size() > static_cast<size_t>(fixture.at("sample_rate").get<int>()) * 30,
                        "full-song fixture must exceed 30 seconds");
                require(output.frames == waveform.size() / fixture.at("inputs_to_logits_ratio").get<size_t>(),
                        "full-song inference silently truncated the waveform");
            }
        }

        bool rejected = false;
        try { model.audio_frames(std::vector<float>(fixture.at("minimum_input_samples").get<size_t>() - 1),
                                 fixture.at("sample_rate")); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "too-short MERT2 waveform must be rejected");
        rejected = false;
        auto invalid = fixture.at("waveform_cases").at(0).at("waveform").get<std::vector<float>>();
        invalid[0] = std::numeric_limits<float>::quiet_NaN();
        try { model.audio_frames(invalid, fixture.at("sample_rate")); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "non-finite MERT2 waveform must be rejected");
        rejected = false;
        try { model.text("MERT2 has no text tower"); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "MERT2 text input must be rejected");
        rejected = false;
        try { model.audio_features({0}, 1); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "MERT2 precomputed feature input must be rejected");
        std::cout << "Native MERT-v2-FullSong parity passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
