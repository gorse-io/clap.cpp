#include "audio.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace {
clap::Audio tone(int rate, double frequency) {
    clap::Audio audio;
    audio.sample_rate = rate;
    audio.samples.resize(rate / 10);
    for (size_t i = 0; i < audio.samples.size(); ++i)
        audio.samples[i] = static_cast<float>(0.5 * std::sin(2 * std::numbers::pi * frequency * i / rate));
    return audio;
}

TEST(Resample, PreservesDurationAmplitudeAndPhase) {
    for (const auto rates : {std::pair<int, int>{8000, 48000},
                             {16000, 48000},
                             {22050, 48000},
                             {44100, 48000},
                             {96000, 48000},
                             {48000, 24000},
                             {44100, 24000},
                             {16000, 24000}}) {
        const int source = rates.first, target = rates.second;
        SCOPED_TRACE(testing::Message() << source << " -> " << target);
        const auto output = clap::resample_audio(tone(source, 1000), target);
        ASSERT_EQ(target, output.sample_rate);
        ASSERT_EQ(static_cast<size_t>(target / 10), output.samples.size());
        // Linear interpolation + causal LPF is not a zero-phase sinc.
        // Fit sine/cosine coefficients: check frequency, gain and bounded delay
        // independently, without requiring the previous algorithm's phase.
        double sine = 0, cosine = 0;
        const size_t begin = target / 100, end = target * 9 / 100; // whole periods, after startup
        for (size_t i = begin; i < end; ++i) {
            const double phase = 2 * std::numbers::pi * 1000 * i / target;
            sine += output.samples[i] * std::sin(phase);
            cosine += output.samples[i] * std::cos(phase);
        }
        sine *= 2.0 / (end - begin);
        cosine *= 2.0 / (end - begin);
        const double gain = std::hypot(sine, cosine) / 0.5;
        const double phase = std::atan2(cosine, sine);
        double error = 0;
        for (size_t i = begin; i < end; ++i) {
            const double angle = 2 * std::numbers::pi * 1000 * i / target;
            error += std::pow(output.samples[i] - sine * std::sin(angle) - cosine * std::cos(angle), 2);
        }
        const double rmse = std::sqrt(error / (end - begin));
        std::cout << source << " -> " << target << " gain=" << gain << " phase=" << phase << " residual=" << rmse
                  << '\n';
        ASSERT_GT(gain, 0.94);
        ASSERT_LT(gain, 1.01);
        ASSERT_LE(phase, 0.01);
        ASSERT_GT(phase, -std::numbers::pi);
        ASSERT_LT(rmse, 0.002);
    }
}

TEST(Resample, SuppressesFrequenciesAboveTargetNyquist) {
    for (const auto rates : {std::pair<int, int>{48000, 16000}, {96000, 48000}, {48000, 24000}}) {
        SCOPED_TRACE(testing::Message() << rates.first << " -> " << rates.second);
        const auto output = clap::resample_audio(tone(rates.first, rates.second * 0.75), rates.second);
        ASSERT_GT(output.samples.size(), 256u);
        double energy = 0;
        for (size_t i = 128; i < output.samples.size() - 128; ++i)
            energy += output.samples[i] * output.samples[i];
        const double rms = std::sqrt(energy / (output.samples.size() - 256));
        std::cout << rates.first << " -> " << rates.second << " alias RMS=" << rms << '\n';
        // At least 35 dB attenuation relative to the 0.5-amplitude input tone.
        ASSERT_LT(rms, 0.006);
    }
}

TEST(Resample, SameRateOutputIsExact) {
    const auto original = tone(48000, 1000);
    ASSERT_EQ(original.samples, clap::resample_audio(original).samples);
}

TEST(Resample, RoundsDurationUpAndPreservesSettledDcGain) {
    for (int source : {8000, 44100, 96000}) {
        SCOPED_TRACE(source);
        const clap::Audio constant{std::vector<float>(source / 10 + 1, 0.25f), source};
        const auto output = clap::resample_audio(constant);
        const auto expected = (static_cast<unsigned long long>(constant.samples.size()) * 48000 + source - 1) / source;
        ASSERT_EQ(expected, output.samples.size());
        ASSERT_LT(std::abs(output.samples.front()), 1e-6f);
        ASSERT_TRUE(std::all_of(output.samples.begin() + 480, output.samples.end(),
                                [](float x) { return std::abs(x - 0.25f) < 1e-6f; }));
    }
}

TEST(Resample, HandlesSingleSampleClips) {
    for (int source : {8000, 44100, 96000}) {
        SCOPED_TRACE(source);
        const auto short_clip = clap::resample_audio({{0.5f}, source});
        ASSERT_EQ((48000ULL + source - 1) / source, short_clip.samples.size());
        for (float x : short_clip.samples)
            ASSERT_TRUE(std::isfinite(x));
    }
}

TEST(Resample, BoundsCausalImpulseLatency) {
    clap::Audio impulse{std::vector<float>(4800, 0), 16000};
    impulse.samples[100] = 1;
    const auto response = clap::resample_audio(impulse);
    const auto peak = std::max_element(response.samples.begin(), response.samples.end());
    const auto delay = peak - response.samples.begin() - 300;
    std::cout << "16000 -> 48000 impulse peak delay=" << delay << " samples\n";
    ASSERT_GT(delay, 0);
    ASSERT_LT(delay, 64);
}

TEST(Resample, PreservesShortAndFractionalSilence) {
    for (const auto rates : {std::pair<int, int>{44100, 48000}, {48000, 16000}, {96000, 44100}}) {
        for (size_t count : {size_t{1}, size_t{2}, size_t{17}, size_t{4097}}) {
            SCOPED_TRACE(testing::Message() << rates.first << " -> " << rates.second << " count=" << count);
            const auto out = clap::resample_audio({std::vector<float>(count, 0), rates.first}, rates.second);
            ASSERT_EQ((count * rates.second + rates.first - 1) / rates.first, out.samples.size());
            for (float x : out.samples)
                ASSERT_EQ(0, x);
        }
    }
}

TEST(Resample, RejectsInvalidInputs) {
    ASSERT_THROW(clap::resample_audio({{}, 48000}, 48000), std::exception);
    ASSERT_THROW(clap::resample_audio({{0}, 0}, 48000), std::exception);
    ASSERT_THROW(clap::resample_audio({{0}, -1}, 48000), std::exception);
    ASSERT_THROW(clap::resample_audio({{0}, 48000}, 0), std::exception);
    ASSERT_THROW(clap::resample_audio({{0}, 48000}, -1), std::exception);
    ASSERT_THROW(clap::resample_audio({{std::numeric_limits<float>::quiet_NaN()}, 48000}, 48000), std::exception);
    ASSERT_THROW(clap::resample_audio({{std::numeric_limits<float>::infinity()}, 44100}, 48000), std::exception);
    ASSERT_THROW(clap::resample_audio({{0}, 1}, std::numeric_limits<int>::max()), std::exception);
}
} // namespace

namespace {
std::filesystem::path fixture_dir;

TEST(AudioDecode, PreservesWaveformsAndMatchesMemoryDecoding) {
    for (const auto &name : {"mono.wav", "mono.flac", "mono.mp3", "stereo.wav", "stereo.flac", "stereo.mp3",
                             "unknown-length.flac", "pcm24.wav", "float.wav"}) {
        SCOPED_TRACE(name);
        const auto audio = clap::decode_audio((fixture_dir / name).string());
        std::ifstream file(fixture_dir / name, std::ios::binary);
        const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
        const auto memory = clap::decode_audio(bytes.data(), bytes.size());
        ASSERT_EQ(audio.sample_rate, memory.sample_rate);
        ASSERT_EQ(audio.samples, memory.samples);
        ASSERT_EQ(48000, audio.sample_rate);
        ASSERT_EQ(4800u, audio.samples.size());
        const bool stereo = std::string(name).find("stereo") == 0;
        double error = 0;
        for (size_t i = 0; i < audio.samples.size(); ++i) {
            ASSERT_TRUE(std::isfinite(audio.samples[i])) << "sample " << i;
            const double left = 12000.0 / 32768 * std::sin(2 * std::numbers::pi * 440 * i / 48000);
            const double right = 12000.0 / 32768 * std::sin(2 * std::numbers::pi * 880 * i / 48000);
            const double expected = stereo ? (left + right) / 2 : left;
            error += std::pow(audio.samples[i] - expected, 2);
        }
        const double rmse = std::sqrt(error / audio.samples.size());
        const double tolerance = std::string(name).find(".mp3") != std::string::npos ? 0.02 : 0.00002;
        ASSERT_LT(rmse, tolerance);
        std::cout << name << " RMSE=" << rmse << '\n';
    }
}

TEST(AudioDecode, PreservesSampleRate) {
    const auto other_rate = clap::decode_audio((fixture_dir / "44100.flac").string());
    ASSERT_EQ(44100, other_rate.sample_rate);
    ASSERT_EQ(4410u, other_rate.samples.size());
}

TEST(AudioDecode, DetectsFormatFromContents) {
    const auto decoded = clap::decode_audio((fixture_dir / "audio.dat").string());
    ASSERT_EQ(4800u, decoded.samples.size());
}

TEST(AudioDecode, MixesSurroundChannelsWithEqualWeights) {
    const auto surround = clap::decode_audio((fixture_dir / "surround.wav").string());
    ASSERT_EQ(4800u, surround.samples.size());
    for (float value : surround.samples)
        ASSERT_LT(std::abs(value - 10500.0f / 32768), 1e-6f);
}

TEST(AudioDecode, RejectsInvalidFilesAndMemoryBuffers) {
    for (const auto &path :
         {fixture_dir / "truncated.wav", fixture_dir / "truncated.flac", fixture_dir / "nonfinite.wav",
          fixture_dir / "missing.wav", fixture_dir / "invalid.bin", fixture_dir / "empty.wav"}) {
        SCOPED_TRACE(path.string());
        ASSERT_THROW(clap::decode_audio(path.string()), std::exception);
        if (std::filesystem::exists(path)) {
            std::ifstream file(path, std::ios::binary);
            const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
            ASSERT_THROW(clap::decode_audio(bytes.data(), bytes.size()), std::exception);
        }
    }
}

TEST(AudioDecode, RejectsNullBuffers) {
    for (size_t size : {size_t{0}, size_t{4}}) {
        SCOPED_TRACE(size);
        ASSERT_THROW(clap::decode_audio(nullptr, size), std::exception);
    }
}
} // namespace

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    if (argc != 2 && !GTEST_FLAG_GET(list_tests)) {
        std::cerr << "usage: clap-audio-test FIXTURE_DIR\n";
        return 1;
    }
    if (argc == 2)
        fixture_dir = argv[1];
    return RUN_ALL_TESTS();
}
