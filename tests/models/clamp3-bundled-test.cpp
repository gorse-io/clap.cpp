#include "clap.h"
#include "../c-api-helpers.h"
#include "audio.h"
#include "feature-test.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace c_api_test;

static void equal(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size())
        throw std::runtime_error("dimension mismatch");
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || std::abs(a[i] - b[i]) > 1e-6f)
            throw std::runtime_error("bundled pipeline differs from dual models");
}
template <class F> static void rejects(F fn) {
    try {
        fn();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("invalid input accepted");
}
int main(int argc, char **argv) {
    try {
        if (argc != 4)
            throw std::runtime_error("bundled, feature-only, MERT models required");
        clap_runtime_options options = clap_default_options();
        options.threads = 2;
        CLAP *raw = nullptr;
        check(clap_create(argv[1], &options, &raw));
        Model bundled(raw, clap_destroy);
        check(clap_create(argv[2], &options, &raw));
        Model target(raw, clap_destroy);
        if (clap_audio_sample_rate(bundled.get()) != 24000)
            throw std::runtime_error("bundled sample rate");
        if (clap_audio_sample_rate(target.get()) != 24000)
            throw std::runtime_error("mandatory frontend sample rate");
        std::vector<float> pcm(48000);
        for (size_t i = 0; i < pcm.size(); ++i)
            pcm[i] = 0.2f * std::sin(i * 0.071f) + 0.1f;
        const auto resampled = clap::resample_audio({pcm, 48000}, 24000).samples;
        const auto features = clap::detail::FeatureTestAccess::extract(argv[3], resampled, 24000);
        const auto expected = clap::detail::FeatureTestAccess::encode(argv[2], features.values, features.frames);
        equal(take_buffer<float>([&](float **out, size_t *count) {
                  const std::vector<float> &api_pcm = pcm;
                  return clap_encode_audio(bundled.get(), api_pcm.data(), api_pcm.size(), 48000, 0, out, count);
              }),
              expected);
        equal(take_buffer<float>([&](float **out, size_t *count) {
                  const std::vector<float> &api_pcm = resampled;
                  return clap_encode_audio(bundled.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
              }),
              expected);
        equal(clap::detail::FeatureTestAccess::encode(argv[1], features.values, features.frames), expected);
        equal(take_buffer<float>([&](float **out, size_t *count) {
                  const std::string &api_text = "ab c";
                  return clap_encode_text(bundled.get(), api_text.c_str(), out, count);
              }),
              take_buffer<float>([&](float **out, size_t *count) {
                  const std::string &api_text = "ab c";
                  return clap_encode_text(target.get(), api_text.c_str(), out, count);
              }));
        rejects([&] {
            take_buffer<float>([&](float **out, size_t *count) {
                const std::vector<float> &api_pcm = std::vector<float>{};
                return clap_encode_audio(bundled.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
            });
        });
        rejects([&] {
            take_buffer<float>([&](float **out, size_t *count) {
                const std::vector<float> &api_pcm = std::vector<float>(23999);
                return clap_encode_audio(bundled.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
            });
        });
        rejects([&] {
            take_buffer<float>([&](float **out, size_t *count) {
                const std::vector<float> &api_pcm = pcm;
                return clap_encode_audio(bundled.get(), api_pcm.data(), api_pcm.size(), 0, 0, out, count);
            });
        });
        rejects([&] {
            take_buffer<float>([&](float **out, size_t *count) {
                const std::vector<float> &api_pcm = pcm;
                return clap_encode_audio(bundled.get(), api_pcm.data(), api_pcm.size(), 24000, 1, out, count);
            });
        });
        rejects([&] {
            take_frames([&](clap_audio_frames *out) {
                const std::vector<float> &api_pcm = pcm;
                return clap_get_audio_frames(bundled.get(), api_pcm.data(), api_pcm.size(), 24000, out);
            });
        });
        pcm[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] {
            take_buffer<float>([&](float **out, size_t *count) {
                const std::vector<float> &api_pcm = pcm;
                return clap_encode_audio(bundled.get(), api_pcm.data(), api_pcm.size(), 24000, 0, out, count);
            });
        });
        std::cout << "Bundled CLaMP3 native waveform/resampling/features/text/invalid-input passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
