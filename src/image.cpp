#ifndef NOMINMAX
#define NOMINMAX
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_MAX_DIMENSIONS 16384
#include "image.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stb_image.h>
#include <stdexcept>

namespace clap {
Image decode_image(const std::string &path) {
    if (path.empty() || path.find('\0') != std::string::npos)
        throw std::invalid_argument("CLAP: nonempty image path without NUL bytes is required");
    std::ifstream stream(std::filesystem::path(std::u8string(path.begin(), path.end())),
                         std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error("CLAP: cannot open image: " + path);
    constexpr size_t max_bytes = size_t{128} << 20;
    constexpr size_t max_pixels = size_t{64} << 20;
    const auto length = stream.tellg();
    if (length <= 0 || static_cast<uintmax_t>(length) > max_bytes)
        throw std::runtime_error("CLAP: image file is empty or exceeds 128 MiB");
    std::vector<uint8_t> data(static_cast<size_t>(length));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size())))
        throw std::runtime_error("CLAP: cannot read image: " + path);
    int width = 0, height = 0, channels = 0;
    const int size = static_cast<int>(data.size());
    if (!stbi_info_from_memory(data.data(), size, &width, &height, &channels) || width <= 0 || height <= 0 ||
        width > STBI_MAX_DIMENSIONS || height > STBI_MAX_DIMENSIONS ||
        static_cast<size_t>(width) > max_pixels / static_cast<size_t>(height))
        throw std::runtime_error("CLAP: invalid image or excessive dimensions");
    const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load_from_memory(data.data(), size, &width, &height, &channels, STBI_rgb), stbi_image_free);
    if (!pixels)
        throw std::runtime_error("CLAP: cannot decode image: " + std::string(stbi_failure_reason()));
    // Revalidate decoder output before computing a byte count, even after header inspection.
    if (width <= 0 || height <= 0 || width > STBI_MAX_DIMENSIONS || height > STBI_MAX_DIMENSIONS ||
        static_cast<size_t>(width) > max_pixels / static_cast<size_t>(height))
        throw std::runtime_error("CLAP: decoded image dimensions exceed limits");
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    return {width, height, {pixels.get(), pixels.get() + bytes}};
}
} // namespace clap
