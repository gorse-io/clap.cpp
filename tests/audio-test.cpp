#include "clap.h"
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
std::filesystem::path fixture_directory;

class AudioDecodeTest : public testing::TestWithParam<const char *> {};

TEST_P(AudioDecodeTest, PreservesFramesAndWaveform) {
    const std::string name = GetParam();
    const auto audio = clap::decode_audio((fixture_directory / name).string());
    ASSERT_EQ(48000, audio.sample_rate);
    ASSERT_EQ(4800u, audio.samples.size());
    const bool stereo = name.find("stereo") == 0;
    double error = 0;
    for (size_t i = 0; i < audio.samples.size(); ++i) {
        ASSERT_TRUE(std::isfinite(audio.samples[i])) << "sample " << i;
        const double left = 12000.0 / 32768 * std::sin(2 * 3.141592653589793 * 440 * i / 48000);
        const double right = 12000.0 / 32768 * std::sin(2 * 3.141592653589793 * 880 * i / 48000);
        const double expected = stereo ? (left + right) / 2 : left;
        error += std::pow(audio.samples[i] - expected, 2);
    }
    const double rmse = std::sqrt(error / audio.samples.size());
    const double tolerance = (name.find(".mp3") != std::string::npos ||
                              name.find(".ogg") != std::string::npos) ? 0.02 : 0.00002;
    EXPECT_LT(rmse, tolerance) << name << " waveform mismatch";
    std::cout << name << " RMSE=" << rmse << '\n';
}

INSTANTIATE_TEST_SUITE_P(Formats, AudioDecodeTest, testing::Values(
    "mono.wav", "mono.flac", "mono.mp3", "stereo.wav", "stereo.flac",
    "stereo.mp3", "mono.ogg", "stereo.ogg", "pcm24.wav", "float.wav"));

TEST(AudioDecode, PreservesNonDefaultSampleRate) {
    for (const char *name : {"44100.flac", "44100.ogg"}) {
        SCOPED_TRACE(name);
        const auto audio = clap::decode_audio((fixture_directory / name).string());
        EXPECT_EQ(44100, audio.sample_rate);
        EXPECT_EQ(4410u, audio.samples.size());
    }
}

TEST(AudioDecode, DetectsFormatFromContents) {
    for (const char *name : {"vorbis.dat", "audio.dat"}) {
        SCOPED_TRACE(name);
        EXPECT_EQ(4800u, clap::decode_audio((fixture_directory / name).string()).samples.size());
    }
}

TEST(AudioDecode, DecodesEveryChainedVorbisLink) {
    const auto audio = clap::decode_audio((fixture_directory / "chained.ogg").string());
    EXPECT_EQ(48000, audio.sample_rate);
    EXPECT_EQ(9600u, audio.samples.size());
}

class InvalidAudioTest : public testing::TestWithParam<const char *> {};
TEST_P(InvalidAudioTest, RejectsInput) {
    EXPECT_THROW(clap::decode_audio((fixture_directory / GetParam()).string()), std::exception);
}
INSTANTIATE_TEST_SUITE_P(InvalidFiles, InvalidAudioTest, testing::Values(
    "missing.wav", "invalid.bin", "empty.wav", "changed-channels.ogg",
    "changed-rate.ogg", "truncated.ogg"));
} // namespace

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    if (argc != 2) {
        std::cerr << "usage: clap-audio-test FIXTURE_DIR\n";
        return 2;
    }
    fixture_directory = argv[1];
    return RUN_ALL_TESTS();
}
