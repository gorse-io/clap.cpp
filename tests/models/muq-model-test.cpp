#include "clap.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
static void compare(const std::vector<float> &actual, const std::vector<float> &expected,
                    float tolerance, const std::string &name) {
    require(actual.size() == expected.size(), name + " size");
    float error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), name + " nonfinite");
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    std::cout << name << " max_abs_error=" << error << '\n';
    require(error <= tolerance, name + " parity");
}
int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: muq-model-test MODEL.gguf FIXTURE.json");
        nlohmann::json fixture;
        std::ifstream input(argv[2]); input >> fixture;
        clap::CLAP model(argv[1], 2);
        require(model.audio_sample_rate() == 24000, "MuQ rate");
        require(model.projection_dimension() == fixture.at("hidden_size"), "MuQ width");
        const float tolerance = fixture.at("tolerance");
        for (const auto &test : fixture.at("waveform_cases")) {
            const auto wave = test.at("waveform").get<std::vector<float>>();
            const auto sequence = model.audio_frames(wave, 24000);
            require(sequence.frames == test.at("frames"), "MuQ frames");
            require(sequence.frames == static_cast<int>((wave.size() / 240 + 3) / 4), "MuQ ceil subsampling");
            require(sequence.dimension == fixture.at("hidden_size"), "MuQ frame width");
            // F16 is storage only for MuQ: silence isolates arithmetic from FFT
            // noise-floor sensitivity and catches unintended activation packing.
            const float bound = test.at("name") == "silence" ? std::min(tolerance, 1e-5f) : tolerance;
            compare(sequence.values, test.at("expected_frames").get<std::vector<float>>(), bound, test.at("name"));
            compare(model.encode_audio(wave, 24000), test.at("expected_pooled").get<std::vector<float>>(), bound, "MuQ pooled");
        }
        const auto wave = fixture.at("waveform_cases").at(1).at("waveform").get<std::vector<float>>();
        const auto resampled = clap::resample_audio({wave, 48000}, 24000);
        compare(model.audio_frames(wave, 48000).values, model.audio_frames(resampled.samples, 24000).values,
                0.0f, "MuQ automatic resampling");
        auto rejects = [&](auto action, const std::string &message) {
            bool rejected = false;
            try { action(); } catch (const std::exception &) { rejected = true; }
            require(rejected, message);
        };
        rejects([&] { model.encode_text("no text tower"); }, "MuQ text must fail");
        rejects([&] { model.audio_frames(std::vector<float>(1024), 24000); }, "MuQ short waveform must fail");
        rejects([&] { model.audio_frames(wave, 0); }, "MuQ invalid rate must fail");
        rejects([&] { model.encode_audio(wave, 24000, 1); }, "MuQ crop must fail");
        auto invalid = wave; invalid[0] = std::numeric_limits<float>::infinity();
        rejects([&] { model.audio_frames(invalid, 24000); }, "MuQ nonfinite must fail");
        std::cout << "MuQ native parity passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
