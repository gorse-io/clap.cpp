#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace clap {
// Top-left-origin, row-major interleaved RGB bytes. Alpha is discarded, not composited.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

// Decode by contents using stb_image. UTF-8 paths, no resizing or orientation changes.
Image decode_image(const std::string &path);
} // namespace clap
