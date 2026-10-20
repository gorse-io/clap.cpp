#include "quantize-output.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static std::string read(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: quantize-output-test DIRECTORY");
        const std::filesystem::path directory(argv[1]);
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        const auto destination = directory / "output.gguf";
        {
            clap::detail::QuantizeOutput output(destination);
            output.write("quantized", 9);
            output.close();
            // Deterministically simulate a competitor winning after temporary
            // creation and completion, immediately before atomic publication.
            std::ofstream(destination, std::ios::binary) << "competitor";
            bool rejected = false;
            try { output.publish(); } catch (const std::exception &) { rejected = true; }
            require(rejected, "concurrent destination was not rejected");
            require(read(destination) == "competitor", "concurrent destination was overwritten");
        }
        require(std::distance(std::filesystem::directory_iterator(directory),
                              std::filesystem::directory_iterator()) == 1, "temporary output leaked");
        std::filesystem::remove(destination);
        {
            clap::detail::QuantizeOutput output(destination);
            output.write("quantized", 9);
            output.close();
            output.publish();
        }
        require(read(destination) == "quantized", "published content changed");
        require(std::distance(std::filesystem::directory_iterator(directory),
                              std::filesystem::directory_iterator()) == 1, "successful temporary output leaked");
        std::filesystem::remove_all(directory);
        std::cout << "Atomic no-clobber publication and cleanup passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
