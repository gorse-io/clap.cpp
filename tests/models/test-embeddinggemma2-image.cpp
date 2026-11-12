#include "model.h"
#include "models/models.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace clap::detail;
static void check(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void compare(const std::vector<float> &actual, const nlohmann::json &expected, float tol) {
    check(actual.size() == expected.size(), "image oracle shape mismatch");
    float error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        check(std::isfinite(actual[i]), "non-finite image output");
        error = std::max(error, std::abs(actual[i] - expected[i].get<float>()));
    }
    std::cout << "image max_abs_error=" << error << " tolerance=" << tol << '\n';
    check(error < tol, "independent image oracle mismatch");
}
int main(int argc, char **argv) {
    try {
        check(argc >= 4, "usage: image-test TEXT MMPROJ ORACLE [reject]");
        ModelLoader text, vision;
        text.load(argv[1]);
        vision.load(argv[2]);
        if (argc == 5 && std::string(argv[4]) == "--audio-only") {
            EmbeddingGemma2Model model(text, 2, &vision);
            try {
                model.image_embedding(clap::Image{1, 1, {1, 2, 3}});
            } catch (const std::exception &e) {
                check(std::string(e.what()).find("vision mmproj") != std::string::npos, e.what());
                return 0;
            }
            throw std::runtime_error("disabled vision tower accepted an image");
        }
        if (argc == 5) {
            try {
                EmbeddingGemma2Model model(text, 2, &vision);
            } catch (const std::exception &e) {
                check(std::string(e.what()).find(argv[4]) != std::string::npos, e.what());
                return 0;
            }
            throw std::runtime_error("invalid vision model accepted");
        }
        GemmaVisionEncoder encoder(vision, 2);
        EmbeddingGemma2Model model(text, 2, &vision);
        auto options = clap_default_options();
        options.threads = 2;
        options.device = "cpu";
        CLAP *raw = nullptr;
        check(clap_create_with_mmproj(argv[1], argv[2], &options, &raw) == CLAP_OK, clap_last_error());
        std::unique_ptr<CLAP, decltype(&clap_destroy)> public_model(raw, clap_destroy);
        std::ifstream stream(argv[3]);
        nlohmann::json fixture;
        stream >> fixture;
        for (const auto &test : fixture.at("cases")) {
            clap::Image image{test.at("width").get<int>(), test.at("height").get<int>(),
                              test.at("pixels").get<std::vector<uint8_t>>()};
            auto soft = encoder.encode(image);
            compare(soft.data, test.at("soft"), 1e-5f);
            auto actual = model.image_embedding(image);
            compare(actual, test.at("embedding"), 1e-5f);
            check(actual == model.image_embedding(image), "nondeterministic image inference");
            float *output = nullptr;
            size_t count = 0;
            check(clap_encode_image(raw, image.pixels.data(), image.pixels.size(), image.width, image.height, &output,
                                    &count) == CLAP_OK,
                  clap_last_error());
            std::unique_ptr<float, decltype(&clap_free)> allocated(output, clap_free);
            compare(std::vector<float>(output, output + count), test.at("embedding"), 1e-5f);
            check(clap_encode_image_file(raw, test.at("path").get<std::string>().c_str(), &output, &count) == CLAP_OK,
                  clap_last_error());
            std::unique_ptr<float, decltype(&clap_free)> from_file(output, clap_free);
            compare(std::vector<float>(output, output + count), test.at("embedding"), 1e-5f);
        }
        for (const clap::Image &image : {clap::Image{}, clap::Image{1, 1, {}}, clap::Image{-1, 1, {1, 2, 3}}}) {
            bool rejected = false;
            try {
                encoder.encode(image);
            } catch (const std::exception &) {
                rejected = true;
            }
            check(rejected, "invalid RGB image accepted");
        }
        EmbeddingGemma2Model missing(text, 2);
        try {
            missing.image_embedding(clap::Image{1, 1, {1, 2, 3}});
        } catch (const std::exception &e) {
            check(std::string(e.what()).find("mmproj") != std::string::npos, e.what());
            return 0;
        }
        throw std::runtime_error("image without mmproj accepted");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
