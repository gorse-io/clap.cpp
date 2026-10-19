#include "clap.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
constexpr double pi = 3.14159265358979323846;
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
clap::Audio tone(int rate, double frequency) {
    clap::Audio audio;
    audio.sample_rate = rate;
    audio.samples.resize(rate / 10);
    for (size_t i = 0; i < audio.samples.size(); ++i)
        audio.samples[i] = static_cast<float>(0.5 * std::sin(2 * pi * frequency * i / rate));
    return audio;
}
void rejected(const clap::Audio &audio, int rate) {
    bool failed = false;
    try { clap::resample_audio(audio, rate); } catch (const std::exception &) { failed = true; }
    require(failed, "invalid resampling input must be rejected");
}
} // namespace

int main() {
    try {
        for (int source : {8000, 16000, 22050, 44100, 96000}) {
            const auto output = clap::resample_audio(tone(source, 1000));
            require(output.sample_rate == 48000, "default target sample rate");
            require(output.samples.size() == 4800, "duration must be preserved");
            double error = 0;
            for (size_t i = 128; i < output.samples.size() - 128; ++i)
                error += std::pow(output.samples[i] - 0.5 * std::sin(2 * pi * 1000 * i / 48000), 2);
            const double rmse = std::sqrt(error / (output.samples.size() - 256));
            std::cout << source << " -> 48000 RMSE=" << rmse << '\n';
            require(rmse < 0.0002, "resampled tone amplitude/phase mismatch");
        }
        for (const auto rates : {std::pair<int, int>{48000, 16000}, {96000, 48000}}) {
            const auto output = clap::resample_audio(tone(rates.first, rates.second * 0.75), rates.second);
            double energy = 0;
            for (size_t i = 128; i < output.samples.size() - 128; ++i)
                energy += output.samples[i] * output.samples[i];
            const double rms = std::sqrt(energy / (output.samples.size() - 256));
            std::cout << rates.first << " -> " << rates.second << " alias RMS=" << rms << '\n';
            require(rms < 0.002, "downsampling must suppress frequencies above target Nyquist");
        }
        const auto original = tone(48000, 1000);
        require(clap::resample_audio(original).samples == original.samples, "same-rate output must be exact");
        for (int source : {8000, 44100, 96000}) {
            const clap::Audio constant{std::vector<float>(101, 0.25f), source};
            const auto output = clap::resample_audio(constant);
            const auto expected = (101ULL * 48000 + source - 1) / source;
            require(output.samples.size() == expected, "fractional duration rounds up");
            require(std::all_of(output.samples.begin(), output.samples.end(), [](float x) {
                return std::abs(x - 0.25f) < 1e-6f;
            }), "DC gain must remain unity, including at clip boundaries");
        }
        require(clap::resample_audio({{0.5f}, 16000}).samples == std::vector<float>(3, 0.5f),
                "single-sample clip");
        rejected({{}, 48000}, 48000);
        rejected({{0}, 0}, 48000);
        rejected({{0}, -1}, 48000);
        rejected({{0}, 48000}, 0);
        rejected({{0}, 48000}, -1);
        rejected({{std::numeric_limits<float>::quiet_NaN()}, 48000}, 48000);
        rejected({{std::numeric_limits<float>::infinity()}, 44100}, 48000);
        rejected({{0}, 1}, std::numeric_limits<int>::max());
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
