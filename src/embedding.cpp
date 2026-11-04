#include "embedding.h"
#include "clap.h"
#include "audio.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Vector = std::vector<float>;
using nlohmann::json;

const char *c_string(const std::string &value) {
    if (value.find('\0') != std::string::npos)
        throw std::invalid_argument("text and paths must contain no NUL bytes");
    return value.c_str();
}

void check(clap_status status) {
    if (status == CLAP_OK)
        return;
    const std::string message = clap_last_error();
    if (status == CLAP_INVALID_ARGUMENT)
        throw std::invalid_argument(message);
    if (status == CLAP_OUT_OF_MEMORY)
        throw std::bad_alloc();
    throw std::runtime_error(message);
}

template <typename T> std::vector<T> take_buffer(T *data, std::size_t count) {
    const std::unique_ptr<T, decltype(&clap_free)> owned(data, clap_free);
    if (!count)
        return {};
    return {data, data + count};
}

// Native projections (notably CLaMP3) are not necessarily unit normalized.
double cosine(const Vector &left, const Vector &right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("Embedding dimensions do not match");
    }
    double dot = 0, left_norm = 0, right_norm = 0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        dot += static_cast<double>(left[i]) * right[i];
        left_norm += static_cast<double>(left[i]) * left[i];
        right_norm += static_cast<double>(right[i]) * right[i];
    }
    if (left_norm == 0 || right_norm == 0)
        return 0;
    return std::clamp(dot / (std::sqrt(left_norm) * std::sqrt(right_norm)), -1.0, 1.0);
}

void write_embeddings(const std::vector<Vector> &vectors, const std::string &format) {
    if (format.empty()) {
        // Native --text/--audio compatibility: exactly one flat JSON vector.
        std::cout << json(vectors.at(0)).dump() << '\n';
    } else if (format == "raw") {
        std::cout << std::fixed << std::setprecision(7);
        for (const auto &vector : vectors) {
            for (std::size_t i = 0; i < vector.size(); ++i) {
                if (i)
                    std::cout << ' ';
                std::cout << vector[i];
            }
            std::cout << '\n';
        }
    } else if (format == "array") {
        std::cout << json(vectors).dump() << '\n';
    } else {
        auto data = json::array();
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            data.push_back({{"object", "embedding"}, {"index", i}, {"embedding", vectors[i]}});
        }
        json result = {{"object", "list"}, {"data", data}};
        if (format == "json+" && vectors.size() > 1) {
            auto similarities = json::array();
            for (const auto &left : vectors) {
                auto row = json::array();
                for (const auto &right : vectors)
                    row.push_back(cosine(left, right));
                similarities.push_back(row);
            }
            result["cosineSimilarity"] = similarities;
        }
        std::cout << result.dump() << '\n';
    }
}
} // namespace

int run_embedding(const EmbeddingOptions &options) {
    try {
        // Preserve native CLI error precedence: decode audio before loading weights.
        clap::Audio decoded;
        if (!options.audio_path.empty())
            decoded = clap::decode_audio(options.audio_path);
        auto runtime = clap_default_options();
        runtime.threads = options.threads;
        runtime.device = c_string(options.device);
        CLAP *handle = nullptr;
        check(clap_create(c_string(options.model_path), &runtime, &handle));
        const std::unique_ptr<CLAP, decltype(&clap_destroy)> model(handle, clap_destroy);
        std::vector<Vector> vectors;
        if (!options.audio_path.empty()) {
            int sample_rate = 0;
            check(clap_audio_sample_rate(model.get(), &sample_rate));
            if (options.crop_requested && sample_rate != 48000) {
                throw std::runtime_error("--crop-offset requires a CLAP model");
            }
            if (options.frames) {
                clap_audio_frames sequence{};
                check(clap_get_audio_frames(model.get(), decoded.samples.data(), decoded.samples.size(),
                                            decoded.sample_rate, &sequence));
                const std::unique_ptr<clap_audio_frames, decltype(&clap_free_audio_frames)> owned(
                    &sequence, clap_free_audio_frames);
                const Vector values =
                    sequence.frames && sequence.dimension
                        ? Vector(sequence.values, sequence.values + sequence.frames * sequence.dimension)
                        : Vector{};
                std::cout
                    << json{{"frames", sequence.frames}, {"dimension", sequence.dimension}, {"values", values}}.dump()
                    << '\n';
            } else {
                float *values = nullptr;
                std::size_t count = 0;
                check(clap_encode_audio(model.get(), decoded.samples.data(), decoded.samples.size(),
                                        decoded.sample_rate, options.crop_offset, &values, &count));
                vectors.push_back(take_buffer(values, count));
            }
        } else if (options.single_text) {
            float *values = nullptr;
            std::size_t count = 0;
            check(clap_encode_text(model.get(), c_string(options.text), &values, &count));
            vectors.push_back(take_buffer(values, count));
        } else {
            // Preserve empty and trailing separator-delimited prompt/file segments.
            std::size_t start = 0;
            while (true) {
                const auto end = options.text.find(options.separator, start);
                const auto text = options.text.substr(start, end == std::string::npos ? end : end - start);
                float *values = nullptr;
                std::size_t count = 0;
                check(clap_encode_text(model.get(), c_string(text), &values, &count));
                vectors.push_back(take_buffer(values, count));
                if (end == std::string::npos)
                    break;
                start = end + options.separator.size();
            }
        }
        if (!vectors.empty()) {
            const std::string format =
                options.format.empty() && !options.single_text && options.audio_path.empty() ? "raw" : options.format;
            write_embeddings(vectors, format);
        }
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("Failed to write embedding output");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
