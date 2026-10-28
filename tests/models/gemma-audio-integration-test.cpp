#include "../c-api-helpers.h"
#include "audio.h"
#include "clap.h"
using namespace c_api_test;
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
int main(int argc, char **argv) {
    if (argc != 4)
        return 1;
    try {
        nlohmann::json fixture;
        std::ifstream(argv[3]) >> fixture;
        const auto pcm = fixture.at("pcm").get<std::vector<float>>();
        const auto expected = fixture.at("embedding").get<std::vector<float>>();
        auto options = clap_default_options();
        options.threads = 2;
        CLAP *raw = nullptr;
        check(clap_create_with_mmproj(argv[1], argv[2], &options, &raw));
        Model model(raw, clap_destroy);
        require(clap_audio_sample_rate(model.get()) == 16000, "wrong audio sample rate");
        const auto actual = take_buffer<float>([&](float **out, size_t *count) {
            return clap_encode_audio(model.get(), pcm.data(), pcm.size(), 16000, 0, out, count);
        });
        require(actual.size() == expected.size(), "wrong audio embedding dimension");
        float maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]), "non-finite audio embedding");
            maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
        }
        std::cout << "public audio scalar oracle max_abs=" << maximum << '\n';
        require(maximum < 5e-5f, "public audio scalar oracle mismatch");
        require(actual == take_buffer<float>([&](float **out, size_t *count) {
                    return clap_encode_audio(model.get(), pcm.data(), pcm.size(), 16000, 0, out, count);
                }),
                "audio output changed across calls");
        const auto resampled = clap::resample_audio({pcm, 32000}, 16000);
        require(take_buffer<float>([&](float **out, size_t *count) {
                    return clap_encode_audio(model.get(), pcm.data(), pcm.size(), 32000, 0, out, count);
                }) == take_buffer<float>([&](float **out, size_t *count) {
                    return clap_encode_audio(model.get(), resampled.samples.data(), resampled.samples.size(), 16000, 0,
                                             out, count);
                }),
                "automatic resampling differs");
        bool rejected = false;
        try {
            take_buffer<float>([&](float **out, size_t *count) {
                return clap_encode_audio(model.get(), pcm.data(), pcm.size(), 16000, 1, out, count);
            });
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "crop offset accepted");
        rejected = false;
        try {
            take_frames([&](clap_audio_frames *out) {
                return clap_get_audio_frames(model.get(), pcm.data(), pcm.size(), 16000, out);
            });
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "frame API unexpectedly accepted");
        rejected = false;
        try {
            take_buffer<float>([&](float **out, size_t *count) {
                return clap_encode_audio(model.get(), pcm.data(), pcm.size(), 0, 0, out, count);
            });
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "zero sample rate accepted");
        for (const auto &invalid : {std::vector<float>{}, std::vector<float>(160, 0),
                                    std::vector<float>(320, std::numeric_limits<float>::quiet_NaN())}) {
            rejected = false;
            try {
                take_buffer<float>([&](float **out, size_t *count) {
                    return clap_encode_audio(model.get(), invalid.data(), invalid.size(), 16000, 0, out, count);
                });
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected, "invalid waveform accepted");
        }
        CLAP *text_raw = nullptr;
        check(clap_create(argv[1], &options, &text_raw));
        Model text_only(text_raw, clap_destroy);
        rejected = false;
        try {
            take_buffer<float>([&](float **out, size_t *count) {
                return clap_encode_audio(text_only.get(), pcm.data(), pcm.size(), 16000, 0, out, count);
            });
        } catch (const std::exception &e) {
            rejected = std::string(e.what()).find("mmproj") != std::string::npos;
        }
        require(rejected, "missing mmproj was not explained");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
    return 0;
}
