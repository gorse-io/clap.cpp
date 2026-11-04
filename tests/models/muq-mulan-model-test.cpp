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

void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
void compare(const std::vector<float> &a, const std::vector<float> &b, float tolerance, const std::string &name) {
    require(a.size() == b.size(), name + " size");
    float error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), name + " finite");
        error = std::max(error, std::abs(a[i] - b[i]));
    }
    std::cout << name << " max_abs_error=" << error << std::endl;
    require(error <= tolerance, name + " parity");
}
int main(int argc, char **argv) {
    try {
        for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            for (bool reference : {false, true}) {
                bool rejected = false;
                try {
                    compare(reference ? std::vector<float>{0, 0} : std::vector<float>{0, bad},
                            reference ? std::vector<float>{0, bad} : std::vector<float>{0, 0}, 0,
                            "nonfinite regression");
                } catch (const std::exception &) {
                    rejected = true;
                }
                require(rejected, "comparison must reject nonfinite native AND reference components");
            }
        }
        require(argc == 3, "usage: muq-mulan-model-test MODEL FIXTURE");
        nlohmann::json f;
        std::ifstream input(argv[2]);
        input >> f;
        clap_runtime_options options = clap_default_options();
        options.threads = 2;
        CLAP *raw = nullptr;
        check(clap_create(argv[1], &options, &raw));
        Model m(raw, clap_destroy);
        require(take_int([&](int *out) { return clap_projection_dimension(m.get(), out); }) == f.at("dimension"),
                "joint width");
        require(take_int([&](int *out) { return clap_audio_sample_rate(m.get(), out); }) == 24000, "rate");
        const float tol = f.at("tolerance");
        for (const auto &c : f.at("texts")) {
            const auto ids = c.at("ids").get<std::vector<int>>();
            require(take_buffer<int>([&](int **out, size_t *count) {
                        const std::string &api_text = c.at("text");
                        return clap_tokenize(m.get(), api_text.c_str(), 10000, out, count);
                    }) == ids,
                    "token parity");
            compare(take_buffer<float>([&](float **out, size_t *count) {
                        const std::vector<int> &api_ids = ids;
                        return clap_encode_tokens(m.get(), api_ids.data(), api_ids.size(), out, count);
                    }),
                    c.at("expected"), tol, "IDs: " + c.at("text").get<std::string>());
            compare(take_buffer<float>([&](float **out, size_t *count) {
                        const std::string &api_text = c.at("text").get<std::string>();
                        return clap_encode_text(m.get(), api_text.c_str(), out, count);
                    }),
                    c.at("expected"), tol, "UTF8");
        }
        for (const auto &c : f.at("audio"))
            compare(take_buffer<float>([&](float **out, size_t *count) {
                        const std::vector<float> &api_pcm = c.at("wave");
                        return clap_encode_audio(m.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
                    }),
                    c.at("expected"), tol, c.at("name"));
        auto reject = [&](auto fn, const std::string &name) {
            bool caught = false;
            try {
                fn();
            } catch (const std::exception &) {
                caught = true;
            }
            require(caught, "must reject " + name);
        };
        const auto wave = f.at("audio").back().at("wave").get<std::vector<float>>();
        compare(take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<float> &api_pcm = wave;
                    return clap_encode_audio(m.get(), api_pcm.data(), api_pcm.size(), 48000, 0, out, count);
                }),
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<float> &api_pcm = clap::resample_audio({wave, 48000}, 24000).samples;
                    return clap_encode_audio(m.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
                }),
                0, "automatic native resampling");
        reject(
            [&] {
                take_frames([&](clap_audio_frames *out) {
                    const std::vector<float> &api_pcm = wave;
                    return clap_get_audio_frames(m.get(), api_pcm.data(), api_pcm.size(), 24000, out);
                });
            },
            "frames");
        reject(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<float> &api_pcm = wave;
                    return clap_encode_audio(m.get(), api_pcm.data(), api_pcm.size(), 24000, 1, out, count);
                });
            },
            "crop");
        reject(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<float> &api_pcm = wave;
                    return clap_encode_audio(m.get(), api_pcm.data(), api_pcm.size(), 0, 0, out, count);
                });
            },
            "rate");
        reject(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<float> &api_pcm = std::vector<float>{};
                    return clap_encode_audio(m.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
                });
            },
            "empty");
        reject(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<float> &api_pcm = std::vector<float>(512);
                    return clap_encode_audio(m.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
                });
            },
            "short");
        auto bad = wave;
        bad[0] = std::numeric_limits<float>::quiet_NaN();
        reject(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<float> &api_pcm = bad;
                    return clap_encode_audio(m.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
                });
            },
            "nonfinite");
        reject(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<int> &api_ids = std::vector<int>{};
                    return clap_encode_tokens(m.get(), api_ids.data(), api_ids.size(), out, count);
                });
            },
            "empty IDs");
        reject(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<int> &api_ids = std::vector<int>{-1};
                    return clap_encode_tokens(m.get(), api_ids.data(), api_ids.size(), out, count);
                });
            },
            "invalid ID");
        reject(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<int> &api_ids = std::vector<int>(600, 0);
                    return clap_encode_tokens(m.get(), api_ids.data(), api_ids.size(), out, count);
                });
            },
            "over context IDs");
        reject(
            [&] {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::string &api_text = std::string(3000, 'a');
                    return clap_encode_text(m.get(), api_text.c_str(), out, count);
                });
            },
            "over context UTF8");
        std::cout << "MuLan native contracts passed" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
