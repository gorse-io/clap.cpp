#include "clap.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
constexpr double pi = 3.14159265358979323846;
clap::Audio tone(int rate, double frequency) {
    clap::Audio audio;
    audio.sample_rate = rate;
    audio.samples.resize(rate / 10);
    for (size_t i = 0; i < audio.samples.size(); ++i)
        audio.samples[i] = static_cast<float>(0.5 * std::sin(2 * pi * frequency * i / rate));
    return audio;
}

class ResampleToneTest : public testing::TestWithParam<int> {};
TEST_P(ResampleToneTest, PreservesDurationAmplitudeAndPhase) {
    const int source = GetParam();
    const auto output = clap::resample_audio(tone(source, 1000));
    EXPECT_EQ(48000, output.sample_rate);
    ASSERT_EQ(4800u, output.samples.size());
    double error = 0;
    for (size_t i = 128; i < output.samples.size() - 128; ++i)
        error += std::pow(output.samples[i] - 0.5 * std::sin(2 * pi * 1000 * i / 48000), 2);
    const double rmse = std::sqrt(error / (output.samples.size() - 256));
    std::cout << source << " -> 48000 RMSE=" << rmse << '\n';
    EXPECT_LT(rmse, 0.0002);
}
INSTANTIATE_TEST_SUITE_P(SourceRates, ResampleToneTest,
                         testing::Values(8000, 16000, 22050, 44100, 96000));

class ResampleAliasTest : public testing::TestWithParam<std::pair<int, int>> {};
TEST_P(ResampleAliasTest, SuppressesFrequenciesAboveTargetNyquist) {
    const auto rates = GetParam();
    const auto output = clap::resample_audio(tone(rates.first, rates.second * 0.75), rates.second);
    ASSERT_GT(output.samples.size(), 256u);
    double energy = 0;
    for (size_t i = 128; i < output.samples.size() - 128; ++i)
        energy += output.samples[i] * output.samples[i];
    const double rms = std::sqrt(energy / (output.samples.size() - 256));
    std::cout << rates.first << " -> " << rates.second << " alias RMS=" << rms << '\n';
    EXPECT_LT(rms, 0.002);
}
INSTANTIATE_TEST_SUITE_P(Downsampling, ResampleAliasTest,
    testing::Values(std::pair<int, int>{48000, 16000}, std::pair<int, int>{96000, 48000}));

TEST(Resample, SameRateOutputIsExact) {
    const auto original = tone(48000, 1000);
    EXPECT_EQ(original.samples, clap::resample_audio(original).samples);
}

class ResampleConstantTest : public testing::TestWithParam<int> {};
TEST_P(ResampleConstantTest, RoundsDurationUpAndPreservesBoundaryGain) {
    const int source = GetParam();
    const clap::Audio constant{std::vector<float>(101, 0.25f), source};
    const auto output = clap::resample_audio(constant);
    const auto expected = (101ULL * 48000 + source - 1) / source;
    EXPECT_EQ(expected, output.samples.size());
    for (size_t i = 0; i < output.samples.size(); ++i)
        EXPECT_LT(std::abs(output.samples[i] - 0.25f), 1e-6f) << "sample " << i;
}
INSTANTIATE_TEST_SUITE_P(SourceRates, ResampleConstantTest, testing::Values(8000, 44100, 96000));

TEST(Resample, HandlesSingleSampleClip) {
    EXPECT_EQ(std::vector<float>(3, 0.5f), clap::resample_audio({{0.5f}, 16000}).samples);
}

TEST(Resample, RejectsInvalidInputs) {
    EXPECT_THROW(clap::resample_audio({{}, 48000}, 48000), std::exception);
    EXPECT_THROW(clap::resample_audio({{0}, 0}, 48000), std::exception);
    EXPECT_THROW(clap::resample_audio({{0}, -1}, 48000), std::exception);
    EXPECT_THROW(clap::resample_audio({{0}, 48000}, 0), std::exception);
    EXPECT_THROW(clap::resample_audio({{0}, 48000}, -1), std::exception);
    EXPECT_THROW(clap::resample_audio({{std::numeric_limits<float>::quiet_NaN()}, 48000}, 48000), std::exception);
    EXPECT_THROW(clap::resample_audio({{std::numeric_limits<float>::infinity()}, 44100}, 48000), std::exception);
    EXPECT_THROW(clap::resample_audio({{0}, 1}, std::numeric_limits<int>::max()), std::exception);
}
} // namespace

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
