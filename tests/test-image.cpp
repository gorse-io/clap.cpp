#include "image.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("usage: test-image IMAGE");
        const auto image = clap::decode_image(argv[1]);
        std::cout << nlohmann::json{{"width", image.width}, {"height", image.height}, {"pixels", image.pixels}}.dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
