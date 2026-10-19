#include "embedding.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <vector>

int run_embedding(const std::string &prompt, const std::string &separator,
                  const std::string &format) {
    // Match llama.cpp's separator semantics, including empty and trailing segments.
    std::size_t count = 1;
    for (std::size_t pos = 0; (pos = prompt.find(separator, pos)) != std::string::npos;
         pos += separator.size()) {
        ++count;
    }

    if (format == "raw") {
        for (std::size_t i = 0; i < count; ++i) {
            std::cout << "1.0000000 0.0000000 0.0000000\n";
        }
        std::cout.flush();
        if (!std::cout) {
            std::cerr << "Failed to write embedding output\n";
            return 1;
        }
        return 0;
    }

    using nlohmann::json;
    auto data = json::array();
    for (std::size_t i = 0; i < count; ++i) {
        const auto vector = json::array({1.0f, 0.0f, 0.0f});
        if (format == "array") {
            data.push_back(vector);
        } else {
            data.push_back({{"object", "embedding"}, {"index", i}, {"embedding", vector}});
        }
    }
    json result = format == "array" ? data : json{{"object", "list"}, {"data", data}};
    if (format == "json+" && count > 1) {
        // All dummy vectors are identical unit vectors, so every cosine is one.
        result["cosineSimilarity"] = json::array();
        const json row = std::vector<float>(count, 1.0f);
        for (std::size_t i = 0; i < count; ++i) {
            result["cosineSimilarity"].push_back(row);
        }
    }
    std::cout << result.dump() << '\n';
    std::cout.flush();
    if (!std::cout) {
        std::cerr << "Failed to write embedding output\n";
        return 1;
    }
    return 0;
}
