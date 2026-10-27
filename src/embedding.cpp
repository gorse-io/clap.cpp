#include "embedding.h"
#include "clap.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Vector = std::vector<float>;
using nlohmann::json;

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
    if (left_norm == 0 || right_norm == 0) return 0;
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
                if (i) std::cout << ' ';
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
                for (const auto &right : vectors) row.push_back(cosine(left, right));
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
        if (!options.audio_path.empty()) decoded = clap::decode_audio(options.audio_path);
        clap::CLAP model(options.model_path, options.threads);
        std::vector<Vector> vectors;
        if (!options.audio_path.empty()) {
            if (options.crop_requested && model.audio_sample_rate() != 48000) {
                throw std::runtime_error("--crop-offset requires a CLAP model");
            }
            if (options.frames || options.clamp3_features) {
                const auto sequence = options.clamp3_features
                    ? model.clamp3_features(decoded.samples, decoded.sample_rate)
                    : model.audio_frames(decoded.samples, decoded.sample_rate);
                std::cout << json{{"frames", sequence.frames}, {"dimension", sequence.dimension},
                                  {"values", sequence.values}}.dump() << '\n';
            } else {
                vectors.push_back(model.encode_audio(decoded.samples, decoded.sample_rate, options.crop_offset));
            }
        } else if (options.single_text) {
            vectors.push_back(model.encode_text(options.text));
        } else {
            // Preserve empty and trailing separator-delimited prompt/file segments.
            std::size_t start = 0;
            while (true) {
                const auto end = options.text.find(options.separator, start);
                vectors.push_back(model.encode_text(options.text.substr(start, end == std::string::npos
                                                                        ? end : end - start)));
                if (end == std::string::npos) break;
                start = end + options.separator.size();
            }
        }
        if (!vectors.empty()) {
            const std::string format = options.format.empty() && !options.single_text && options.audio_path.empty()
                ? "raw" : options.format;
            write_embeddings(vectors, format);
        }
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("Failed to write embedding output");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
