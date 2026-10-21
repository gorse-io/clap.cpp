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
        for (const auto rates : {std::pair<int, int>{8000, 48000}, {16000, 48000},
                                {22050, 48000}, {44100, 48000}, {96000, 48000},
                                {48000, 24000}, {44100, 24000}, {16000, 24000}}) {
            const int source = rates.first, target = rates.second;
            const auto output = clap::resample_audio(tone(source, 1000), target);
            require(output.sample_rate == target, "target sample rate");
            require(output.samples.size() == static_cast<size_t>(target / 10), "duration must be preserved");
            // Linear interpolation + causal LPF is not a zero-phase sinc.
            // Fit sine/cosine coefficients: check frequency, gain and bounded delay
            // independently, without requiring the previous algorithm's phase.
            double sine = 0, cosine = 0;
            const size_t begin = target / 100, end = target * 9 / 100; // whole periods, after startup
            for (size_t i = begin; i < end; ++i) {
                const double phase = 2 * pi * 1000 * i / target;
                sine += output.samples[i] * std::sin(phase);
                cosine += output.samples[i] * std::cos(phase);
            }
            sine *= 2.0 / (end - begin);
            cosine *= 2.0 / (end - begin);
            const double gain = std::hypot(sine, cosine) / 0.5;
            const double phase = std::atan2(cosine, sine);
            double error = 0;
            for (size_t i = begin; i < end; ++i) {
                const double angle = 2 * pi * 1000 * i / target;
                error += std::pow(output.samples[i] - sine * std::sin(angle) - cosine * std::cos(angle), 2);
            }
            const double rmse = std::sqrt(error / (end - begin));
            std::cout << source << " -> " << target << " gain=" << gain << " phase=" << phase << " residual=" << rmse << '\n';
            require(gain > 0.94 && gain < 1.01, "resampled tone amplitude mismatch");
            require(phase <= 0.01 && phase > -pi, "resampled tone delay out of bounds");
            require(rmse < 0.002, "resampled tone frequency/distortion mismatch");
        }
        for (const auto rates : {std::pair<int, int>{48000, 16000}, {96000, 48000}, {48000, 24000}}) {
            const auto output = clap::resample_audio(tone(rates.first, rates.second * 0.75), rates.second);
            double energy = 0;
            for (size_t i = 128; i < output.samples.size() - 128; ++i)
                energy += output.samples[i] * output.samples[i];
            const double rms = std::sqrt(energy / (output.samples.size() - 256));
            std::cout << rates.first << " -> " << rates.second << " alias RMS=" << rms << '\n';
            // At least 35 dB attenuation relative to the 0.5-amplitude input tone.
            require(rms < 0.006, "downsampling must suppress frequencies above target Nyquist");
        }
        const auto original = tone(48000, 1000);
        require(clap::resample_audio(original).samples == original.samples, "same-rate output must be exact");
        for (int source : {8000, 44100, 96000}) {
            const clap::Audio constant{std::vector<float>(source / 10 + 1, 0.25f), source};
            const auto output = clap::resample_audio(constant);
            const auto expected = (static_cast<unsigned long long>(constant.samples.size()) * 48000 + source - 1) / source;
            require(output.samples.size() == expected, "fractional duration rounds up");
            require(std::abs(output.samples.front()) < 1e-6f, "causal DC startup");
            require(std::all_of(output.samples.begin() + 480, output.samples.end(), [](float x) {
                return std::abs(x - 0.25f) < 1e-6f;
            }), "settled DC gain must remain unity through end of clip");
        }
        for (int source : {8000, 44100, 96000}) {
            const auto short_clip = clap::resample_audio({{0.5f}, source});
            require(short_clip.samples.size() == (48000ULL + source - 1) / source, "single-sample duration");
            for (float x : short_clip.samples) require(std::isfinite(x), "finite single-sample output");
        }
        clap::Audio impulse{std::vector<float>(4800, 0), 16000};
        impulse.samples[100] = 1;
        const auto response = clap::resample_audio(impulse);
        const auto peak = std::max_element(response.samples.begin(), response.samples.end());
        const auto delay = peak - response.samples.begin() - 300;
        std::cout << "16000 -> 48000 impulse peak delay=" << delay << " samples\n";
        require(delay > 0 && delay < 64, "causal resampler impulse latency");
        for (const auto rates : {std::pair<int, int>{44100, 48000}, {48000, 16000}, {96000, 44100}}) {
            for (size_t count : {size_t{1}, size_t{2}, size_t{17}, size_t{4097}}) {
                const auto out = clap::resample_audio({std::vector<float>(count, 0), rates.first}, rates.second);
                require(out.samples.size() == (count * rates.second + rates.first - 1) / rates.first,
                        "short/fractional duration");
                for (float x : out.samples) require(x == 0, "silence stays silent");
            }
        }
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
