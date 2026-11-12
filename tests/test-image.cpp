#include "image.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
int wmain(int argc, wchar_t **argv) {
#else
int main(int argc, char **argv) {
#endif
    try {
        if (argc != 2)
            throw std::runtime_error("usage: test-image IMAGE");
        const auto path = std::filesystem::path(argv[1]).u8string();
        const auto image = clap::decode_image(std::string(path.begin(), path.end()));
        std::cout << nlohmann::json{{"width", image.width}, {"height", image.height}, {"pixels", image.pixels}}.dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
