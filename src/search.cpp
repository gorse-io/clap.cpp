#include "audio.h"
#include "clap.h"
#include "search.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;

const char *c_string(const std::string &value) {
    if (value.find('\0') != std::string::npos)
        throw std::invalid_argument("text and paths must contain no NUL bytes");
    return value.c_str();
}

std::string utf8_path(const fs::path &path) {
    const auto value = path.generic_u8string();
    return {value.begin(), value.end()};
}

void check(clap_status status) {
    if (status == CLAP_OK)
        return;
    if (status == CLAP_OUT_OF_MEMORY)
        throw std::bad_alloc();
    throw std::runtime_error(clap_last_error());
}

using Embedding = std::vector<double>;

Embedding normalize(float *data, std::size_t count) {
    const std::unique_ptr<float, decltype(&clap_free)> owned(data, clap_free);
    if (!data || !count)
        throw std::runtime_error("Search received an empty embedding");
    double norm_squared = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i]))
            throw std::runtime_error("Search received a non-finite embedding");
        norm_squared += static_cast<double>(data[i]) * data[i];
    }
    if (!std::isfinite(norm_squared) || norm_squared <= 0)
        throw std::runtime_error("Search received an invalid or zero-norm embedding");
    const double norm = std::sqrt(norm_squared);
    Embedding result(count);
    for (std::size_t i = 0; i < count; ++i)
        result[i] = data[i] / norm;
    return result;
}

Embedding encode_query(CLAP *model, const SearchOptions &options) {
    float *values = nullptr;
    std::size_t count = 0;
    if (options.text_requested)
        check(clap_encode_text(model, c_string(options.text), &values, &count));
    else
        check(clap_encode_audio_file(model, c_string(options.audio_path), 0, &values, &count));
    return normalize(values, count);
}

bool audio_file(const fs::path &path) {
    auto extension = utf8_path(path.extension());
    for (auto &c : extension)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c + ('a' - 'A'));
    return extension == ".wav" || extension == ".mp3" || extension == ".flac";
}

struct Result {
    std::string path;
    double score;
};

struct Better {
    bool operator()(const Result &left, const Result &right) const {
        return left.score > right.score || (left.score == right.score && left.path < right.path);
    }
};
} // namespace

int cmd_search(const SearchOptions &options) {
    try {
        if (!options.top_k)
            throw std::invalid_argument("--top-k must be positive");
        const fs::path directory = fs::u8path(c_string(options.directory));
        if (!fs::is_directory(directory))
            throw std::invalid_argument("Search directory does not exist or is not a directory: " + options.directory);
        if (fs::is_symlink(fs::symlink_status(directory)))
            throw std::invalid_argument("Search directory must not be a directory symlink: " + options.directory);

        auto runtime = clap_default_options();
        runtime.threads = options.threads;
        runtime.device = c_string(options.device);
        CLAP *handle = nullptr;
        check(clap_create(c_string(options.model_path), &runtime, &handle));
        const std::unique_ptr<CLAP, decltype(&clap_destroy)> model(handle, clap_destroy);
        const auto query = encode_query(model.get(), options);

        // Keep only the best k scores, not the corpus embeddings. Better puts the
        // worst retained result at the top so replacement is O(log k).
        std::priority_queue<Result, std::vector<Result>, Better> best;
        // The default iterator does not follow directory symlinks. Traversal
        // errors are fatal rather than returning an unexplained partial search.
        for (const auto &entry : fs::recursive_directory_iterator(directory)) {
            if (!entry.is_regular_file() || !audio_file(entry.path()))
                continue;
            const auto path = utf8_path(entry.path());
            float *values = nullptr;
            std::size_t count = 0;
            const auto status = clap_encode_audio_file(model.get(), c_string(path), 0, &values, &count);
            if (status != CLAP_OK) {
                if (status != CLAP_ERROR)
                    check(status);
                const std::string error = clap_last_error();
                // The C API uses CLAP_ERROR for both decoder and model errors.
                // Re-probe decoding only after failure to distinguish them without
                // decoding successful candidates twice or hiding model failures.
                try {
                    (void)clap::decode_audio(path);
                } catch (const std::runtime_error &decode_error) {
                    std::cerr << "Warning: skipping " << path << ": " << decode_error.what() << '\n';
                    continue;
                }
                throw std::runtime_error("Failed to encode " + path + ": " + error);
            }
            const auto candidate = normalize(values, count);
            if (candidate.size() != query.size())
                throw std::runtime_error("Search query and audio embedding dimensions do not match");
            double score = 0;
            for (std::size_t i = 0; i < query.size(); ++i)
                score += query[i] * candidate[i];
            if (!std::isfinite(score))
                throw std::runtime_error("Search produced a non-finite cosine similarity");
            Result result{path, std::clamp(score, -1.0, 1.0)};
            if (best.size() < options.top_k) {
                best.push(std::move(result));
            } else if (Better{}(result, best.top())) {
                best.pop();
                best.push(std::move(result));
            }
        }
        if (best.empty())
            throw std::runtime_error("No searchable audio files found in: " + options.directory);
        std::vector<Result> results;
        while (!best.empty()) {
            results.push_back(best.top());
            best.pop();
        }
        std::sort(results.begin(), results.end(), Better{});
        auto output = nlohmann::json::array();
        for (const auto &result : results)
            output.push_back({{"path", result.path}, {"score", result.score}});
        std::cout << output.dump() << '\n';
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("Failed to write search output");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
