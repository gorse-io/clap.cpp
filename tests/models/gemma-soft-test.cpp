#include "models/models.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>

namespace clap::detail {
struct FeatureTestAccess {
    static std::vector<float> soft(const EmbeddingGemma2Model &model, Matrix x, const std::vector<int> &mask) {
        return model.encode(std::move(x), mask);
    }
};
}
int main(int argc, char **argv) {
    if (argc != 3) return 1;
    try {
        clap::detail::ModelLoader loader; loader.load(argv[1]);
        clap::detail::EmbeddingGemma2Model model(loader, 2);
        nlohmann::json fixture; std::ifstream(argv[2]) >> fixture;
        for (const auto &test : fixture.at("soft_cases")) {
            const auto rows = test.at("embeddings").get<std::vector<std::vector<float>>>();
            clap::detail::Matrix x(static_cast<int>(rows.size()), static_cast<int>(rows[0].size()));
            for (size_t r = 0; r < rows.size(); ++r) std::copy(rows[r].begin(), rows[r].end(), x[static_cast<int>(r)]);
            const auto actual = clap::detail::FeatureTestAccess::soft(model, std::move(x), test.at("mask").get<std::vector<int>>());
            const auto expected = test.at("embedding").get<std::vector<float>>();
            if (actual.size() != expected.size()) return 2;
            for (size_t i = 0; i < actual.size(); ++i)
                if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 5e-6f) return 3;
        }
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 4; }
    return 0;
}
