/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "audio.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t limit = 480000;

void expect_length_error(const std::vector<float> &pcm, size_t offset) {
    try {
        (void)clap::log_mel(pcm, 48000, offset);
        FAIL() << "Overlength CLAP audio was silently cropped";
    } catch (const std::runtime_error &error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("480000"), std::string::npos) << message;
        EXPECT_NE(message.find("10 seconds"), std::string::npos) << message;
    }
}
} // namespace

TEST(AudioLimit, RejectsOneSampleOverLimit) { expect_length_error(std::vector<float>(limit + 1, 0.1f), 0); }

TEST(AudioLimit, CropOffsetCannotBypassLimit) {
    const std::vector<float> pcm(limit + 1, 0.1f);
    expect_length_error(pcm, 1);
    expect_length_error(pcm, std::numeric_limits<size_t>::max());
}

TEST(AudioLimit, AcceptsExactLimit) {
    const auto mel = clap::log_mel(std::vector<float>(limit, 0.1f));
    ASSERT_EQ(mel.size(), 1001u * 64);
    EXPECT_TRUE(std::all_of(mel.begin(), mel.end(), [](float x) { return std::isfinite(x); }));
}

TEST(AudioLimit, PreservesShortClipRepeatPadding) {
    const std::vector<float> short_pcm(48000, 0.1f);
    EXPECT_EQ(clap::log_mel(short_pcm), clap::log_mel(std::vector<float>(limit, 0.1f)));
}

TEST(AudioLimit, RejectsUnsupportedOffsetsOnValidLengthAudio) {
    for (const size_t samples : {size_t{48000}, limit}) {
        try {
            (void)clap::log_mel(std::vector<float>(samples, 0.1f), 48000, 1);
            FAIL() << "Nonzero crop offset was accepted";
        } catch (const std::runtime_error &error) {
            EXPECT_NE(std::string(error.what()).find("unsupported"), std::string::npos);
        }
    }
}

TEST(AudioLimit, ChecksLengthAfterResampling) {
    for (const int rate : {24000, 96000}) {
        const auto exact = clap::resample_audio({std::vector<float>(10 * rate, 0.1f), rate}, 48000);
        ASSERT_EQ(exact.samples.size(), limit);
        EXPECT_NO_THROW((void)clap::log_mel(exact.samples));
        const auto longer = clap::resample_audio({std::vector<float>(10 * rate + 1, 0.1f), rate}, 48000);
        ASSERT_GT(longer.samples.size(), limit);
        expect_length_error(longer.samples, 0);
        expect_length_error(longer.samples, 1);
    }
}
