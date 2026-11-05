#include "clap.h"
#include "../c-api-helpers.h"
#include "audio.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace c_api_test;

static void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void compare(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance,
                    const std::string &name) {
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
        std::ifstream input(argv[2]);
        input >> fixture;
        clap_runtime_options options = clap_default_options();
        options.threads = 2;
        CLAP *raw = nullptr;
        check(clap_create(argv[1], &options, &raw));
        Model model(raw, clap_destroy);
        require(clap_audio_sample_rate(model.get()) == 24000, "MuQ rate");
        require(clap_projection_dimension(model.get()) == fixture.at("hidden_size"), "MuQ width");
        const float tolerance = fixture.at("tolerance");
        for (const auto &test : fixture.at("waveform_cases")) {
            const auto wave = test.at("waveform").get<std::vector<float>>();
            const auto sequence = take_frames([&](clap_audio_frames *out) {
                const std::vector<float> &api_pcm = wave;
                return clap_get_audio_frames(model.get(), api_pcm.data(), api_pcm.size(), 24000, out);
            });
            require(sequence.frames == test.at("frames"), "MuQ frames");
            require(sequence.frames == static_cast<int>((wave.size() / 240 + 3) / 4), "MuQ ceil subsampling");
            require(sequence.dimension == fixture.at("hidden_size"), "MuQ frame width");
            // F16 is storage only for MuQ: silence isolates arithmetic from FFT
            // noise-floor sensitivity and catches unintended activation packing.
            const float bound = test.at("name") == "silence" ? std::min(tolerance, 1e-5f) : tolerance;
            compare(sequence.values, test.at("expected_frames").get<std::vector<float>>(), bound, test.at("name"));
            compare(take_buffer<float>([&](float **out, size_t *count) {
                        const std::vector<float> &api_pcm = wave;
                        return clap_encode_audio(model.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
                    }),
                    test.at("expected_pooled").get<std::vector<float>>(), bound, "MuQ pooled");
        }
        const auto wave = fixture.at("waveform_cases").at(1).at("waveform").get<std::vector<float>>();
        const auto resampled = clap::resample_audio({wave, 48000}, 24000);
        compare(take_frames([&](clap_audio_frames *out) {
                    const std::vector<float> &api_pcm = wave;
                    return clap_get_audio_frames(model.get(), api_pcm.data(), api_pcm.size(), 48000, out);
                }).values,
                take_frames([&](clap_audio_frames *out) {
                    const std::vector<float> &api_pcm = resampled.samples;
                    return clap_get_audio_frames(model.get(), api_pcm.data(), api_pcm.size(), 24000, out);
                }).values,
                0.0f, "MuQ automatic resampling");
        auto rejects = [&](auto action, const std::string &message) {
            bool rejected = false;
            try {
                action();
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected, message);
        };
        rejects(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::string &api_text = "no text tower";
                    return clap_encode_text(model.get(), api_text.c_str(), out, count);
                });
            },
            "MuQ text must fail");
        rejects(
            [&] {
                take_frames([&](clap_audio_frames *out) {
                    const std::vector<float> &api_pcm = std::vector<float>(1024);
                    return clap_get_audio_frames(model.get(), api_pcm.data(), api_pcm.size(), 24000, out);
                });
            },
            "MuQ short waveform must fail");
        rejects(
            [&] {
                take_frames([&](clap_audio_frames *out) {
                    const std::vector<float> &api_pcm = wave;
                    return clap_get_audio_frames(model.get(), api_pcm.data(), api_pcm.size(), 0, out);
                });
            },
            "MuQ invalid rate must fail");
        rejects(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<float> &api_pcm = wave;
                    return clap_encode_audio(model.get(), api_pcm.data(), api_pcm.size(), 24000, 1, out, count);
                });
            },
            "MuQ crop must fail");
        auto invalid = wave;
        invalid[0] = std::numeric_limits<float>::infinity();
        rejects(
            [&] {
                take_frames([&](clap_audio_frames *out) {
                    const std::vector<float> &api_pcm = invalid;
                    return clap_get_audio_frames(model.get(), api_pcm.data(), api_pcm.size(), 24000, out);
                });
            },
            "MuQ nonfinite must fail");
        std::cout << "MuQ native parity passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
