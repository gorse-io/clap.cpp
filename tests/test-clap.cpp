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
#include "clap.h"

#include <gtest/gtest.h>
#include <httplib.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void download_model(const std::filesystem::path &path) {
    if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0) {
        return;
    }
    std::filesystem::create_directories(path.parent_path());
    auto partial = path;
    partial += ".part";
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot open model download: " + partial.string());
    }

    httplib::Client client("https://huggingface.co");
    client.set_follow_location(true);
    client.set_connection_timeout(30);
    client.set_read_timeout(120);
    const auto response = client.Get("/gorse-io/larger_clap_general-GGUF/resolve/main/larger_clap_general-q4_k_m.gguf",
                                     [&](const char *data, size_t size) {
                                         output.write(data, static_cast<std::streamsize>(size));
                                         return static_cast<bool>(output);
                                     });
    output.close();
    if (!response || response->status != 200 || !output) {
        std::filesystem::remove(partial);
        throw std::runtime_error("Model download failed: " + (response ? "HTTP " + std::to_string(response->status)
                                                                       : httplib::to_string(response.error())));
    }
    std::filesystem::rename(partial, path);
}

struct Embedding {
    std::vector<float> values;
    size_t count;

    explicit Embedding(size_t dimension) : values(dimension), count(dimension) {}

    Embedding(const Embedding &) = delete;
    Embedding &operator=(const Embedding &) = delete;
};

float cosine(const Embedding &a, const Embedding &b) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < a.count; ++i) {
        dot += a.values[i] * b.values[i];
        norm_a += a.values[i] * a.values[i];
        norm_b += b.values[i] * b.values[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}
} // namespace

TEST(CLAP, CallerOwnedEmbeddingBuffers) {
    ASSERT_NO_THROW(download_model(CLAP_TEST_MODEL));
    auto options = clap_default_options();
    options.threads = 1;
    CLAP *raw = nullptr;
    ASSERT_EQ(clap_create(CLAP_TEST_MODEL, &options, &raw), CLAP_OK) << clap_last_error();
    const std::unique_ptr<CLAP, decltype(&clap_destroy)> model(raw, clap_destroy);
    const int dimension = clap_projection_dimension(model.get());
    ASSERT_GT(dimension, 0);
    const size_t width = static_cast<size_t>(dimension);
    const float sentinel = 12345.0f;
    std::vector<float> output(width + 3, sentinel);
    const std::vector<float> pcm(4800, 0.0f);
    const auto text = [&](float *p, size_t n) { return clap_encode_text(model.get(), "Sound of a dog", p, n); };
    const auto audio = [&](float *p, size_t n) {
        return clap_encode_audio(model.get(), pcm.data(), pcm.size(), 48000, 0, p, n);
    };
    const auto file = [&](float *p, size_t n) { return clap_encode_audio_file(model.get(), CLAP_TEST_AUDIO, 0, p, n); };
    const auto verify = [&](const auto &encode) {
        EXPECT_EQ(encode(nullptr, width), CLAP_INVALID_ARGUMENT);
        for (size_t capacity : {size_t(0), width - 1}) {
            std::fill(output.begin(), output.end(), sentinel);
            EXPECT_EQ(encode(output.data(), capacity), CLAP_INVALID_ARGUMENT);
            EXPECT_TRUE(std::all_of(output.begin(), output.end(), [&](float v) { return v == sentinel; }));
        }
        EXPECT_EQ(encode(output.data(), width), CLAP_OK) << clap_last_error();
        const std::vector<float> exact(output.begin(), output.begin() + width);
        std::fill(output.begin(), output.end(), sentinel);
        EXPECT_EQ(encode(output.data(), output.size()), CLAP_OK) << clap_last_error();
        for (size_t i = 0; i < width; ++i) {
            EXPECT_TRUE(std::isfinite(output[i]));
            EXPECT_NEAR(output[i], exact[i], 1e-6f);
        }
        for (size_t i = width; i < output.size(); ++i)
            EXPECT_EQ(output[i], sentinel);
    };
    verify(text);
    verify(audio);
    verify(file);
    EXPECT_EQ(clap_encode_text(nullptr, "text", output.data(), output.size()), CLAP_INVALID_ARGUMENT);
}

TEST(CLAP, AudioTextSimilarity) {
    ASSERT_NO_THROW(download_model(CLAP_TEST_MODEL));

    auto options = clap_default_options();
    options.threads = 1;
    CLAP *raw = nullptr;
    const auto status = clap_create(CLAP_TEST_MODEL, &options, &raw);
    const std::unique_ptr<CLAP, decltype(&clap_destroy)> clap(raw, clap_destroy);
    ASSERT_EQ(status, CLAP_OK) << clap_last_error();
    ASSERT_NE(clap, nullptr);

    const auto dimension = static_cast<size_t>(clap_projection_dimension(clap.get()));
    Embedding audio_embedding(dimension);
    Embedding text_embedding_1(dimension);
    Embedding text_embedding_2(dimension);
    ASSERT_EQ(clap_encode_audio_file(clap.get(), CLAP_TEST_AUDIO, 0, audio_embedding.values.data(),
                                     audio_embedding.values.size()),
              CLAP_OK)
        << clap_last_error();
    ASSERT_EQ(
        clap_encode_text(clap.get(), "Sound of a dog", text_embedding_1.values.data(), text_embedding_1.values.size()),
        CLAP_OK)
        << clap_last_error();
    ASSERT_EQ(clap_encode_text(clap.get(), "Sound of vaccum cleaner", text_embedding_2.values.data(),
                               text_embedding_2.values.size()),
              CLAP_OK)
        << clap_last_error();
    ASSERT_GT(audio_embedding.count, 0u);
    ASSERT_EQ(audio_embedding.count, text_embedding_1.count);
    ASSERT_EQ(audio_embedding.count, text_embedding_2.count);
    ASSERT_FALSE(audio_embedding.values.empty());
    ASSERT_FALSE(text_embedding_1.values.empty());
    ASSERT_FALSE(text_embedding_2.values.empty());

    const float distance_1 = cosine(audio_embedding, text_embedding_1);
    const float distance_2 = cosine(audio_embedding, text_embedding_2);
    // softmax(cosine * exp(logit_scale_a)), as in Transformers ClapModel.
    // This is the learned scalar from laion/larger_clap_general's
    // pytorch_model.bin, not the configuration's initialization value.
    const float logit_scale = std::exp(3.6549275f);
    const float logit_1 = distance_1 * logit_scale;
    const float logit_2 = distance_2 * logit_scale;
    const float max_logit = std::max(logit_1, logit_2);
    const float exp_1 = std::exp(logit_1 - max_logit);
    const float exp_2 = std::exp(logit_2 - max_logit);
    const float score_1 = exp_1 / (exp_1 + exp_2);
    const float score_2 = exp_2 / (exp_1 + exp_2);
    EXPECT_GT(score_1, 0.99f);
    EXPECT_LT(score_2, 0.01f);
}
