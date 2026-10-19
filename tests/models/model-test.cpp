#include "clap.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

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
    if (error > tolerance) {
        for (size_t i = 0; i < actual.size(); ++i) if (std::abs(actual[i] - expected[i]) == error) {
            std::cerr << name << " worst index=" << i << " actual=" << actual[i] << " expected=" << expected[i] << '\n';
            break;
        }
    }
    require(error <= tolerance, name + " numerical mismatch");
}
int main(int argc, char **argv) {
    try {
        const auto silent = clap::log_mel(std::vector<float>(480000, 0));
        require(silent.size() == 1001 * 64, "log mel shape");
        compare(silent, std::vector<float>(1001 * 64, -100), 1e-5f, "silence log mel");
        bool rejected = false;
        try { clap::log_mel({}); } catch (const std::exception &) { rejected = true; }
        require(rejected, "empty waveform must be rejected");
        rejected = false;
        try { clap::log_mel({0}, 16000); } catch (const std::exception &) { rejected = true; }
        require(rejected, "incorrect sample rate must be rejected");
        rejected = false;
        try { clap::log_mel({std::numeric_limits<float>::quiet_NaN()}); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "non-finite waveform must be rejected");
        std::vector<float> long_clip(480003, 0); long_clip[24000] = 0.5f;
        const std::vector<float> crop(long_clip.begin() + 2, long_clip.begin() + 480002);
        compare(clap::log_mel(long_clip, 48000, 2), clap::log_mel(crop), 0, "explicit long-clip crop");
        rejected = false;
        try { clap::log_mel(long_clip, 48000, 4); } catch (const std::exception &) { rejected = true; }
        require(rejected, "out-of-range crop must be rejected");
        rejected = false;
        try { clap::Model missing("/this/path/does/not/exist.gguf"); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "missing GGUF must be rejected");
        if (argc == 1) return 0;
        require(argc == 3, "usage: clap-model-test [MODEL.gguf FIXTURE.json]");
        std::ifstream input(argv[2]);
        nlohmann::json fixture; input >> fixture;
        clap::Model model(argv[1], 2);
        const float embedding_tolerance = fixture.value("embedding_tolerance", 5e-5f);
        for (const auto &test : fixture.at("text_cases")) {
            const auto ids = model.tokenize(test.at("text").get<std::string>());
            require(ids == test.at("ids").get<std::vector<int>>(), "tokenizer mismatch: " + test.at("text").get<std::string>());
            compare(model.text(test.at("padded_ids").get<std::vector<int>>()),
                    test.at("expected").get<std::vector<float>>(), embedding_tolerance, "text embedding");
        }
        compare(model.audio_features(fixture.at("features").get<std::vector<float>>(), fixture.at("frames")),
                fixture.at("expected_audio").get<std::vector<float>>(), embedding_tolerance, "audio embedding");
        if (fixture.contains("waveform")) {
            compare(clap::log_mel(fixture.at("waveform").get<std::vector<float>>()),
                    fixture.at("mel").get<std::vector<float>>(), 3e-4f, "waveform log mel");
            if (fixture.value("end_to_end", false))
                compare(model.audio(fixture.at("waveform").get<std::vector<float>>()),
                        fixture.at("expected_audio").get<std::vector<float>>(), 5e-5f, "waveform embedding");
        }
        std::cout << "Native CLAP parity passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
