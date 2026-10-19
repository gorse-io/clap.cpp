#include "models.h"
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
std::uint32_t u32(const unsigned char *p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
           (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
std::uint16_t u16(const unsigned char *p) { return std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8); }
std::vector<float> read_wav(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open WAV: " + path);
    const auto size = input.tellg();
    if (size < 12 || size > 1024 * 1024 * 1024) throw std::runtime_error("Invalid WAV size (maximum 1 GiB)");
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    input.seekg(0); input.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!input || std::memcmp(bytes.data(), "RIFF", 4) || std::memcmp(bytes.data() + 8, "WAVE", 4))
        throw std::runtime_error("Expected little-endian RIFF/WAVE audio");
    unsigned format = 0, bits = 0, channels = 0, rate = 0, block_align = 0;
    size_t offset = 0, length = 0;
    for (size_t pos = 12; pos + 8 <= bytes.size();) {
        const size_t n = u32(bytes.data() + pos + 4), start = pos + 8;
        if (n > bytes.size() - start) throw std::runtime_error("Truncated WAV chunk");
        if (std::memcmp(bytes.data() + pos, "fmt ", 4) == 0) {
            if (n < 16) throw std::runtime_error("Truncated WAV format");
            format = u16(bytes.data() + start); channels = u16(bytes.data() + start + 2);
            rate = u32(bytes.data() + start + 4); block_align = u16(bytes.data() + start + 12);
            bits = u16(bytes.data() + start + 14);
        } else if (std::memcmp(bytes.data() + pos, "data", 4) == 0) { offset = start; length = n; }
        pos = start + n + (n % 2);
    }
    if (rate != 48000 || channels != 1) throw std::runtime_error("WAV must be mono 48000 Hz; resample/downmix before inference");
    if (!((format == 1 && bits == 16) || (format == 3 && bits == 32)))
        throw std::runtime_error("WAV must contain PCM16 or IEEE float32 samples");
    const unsigned width = bits / 8;
    if (!length || length % width || block_align != width) throw std::runtime_error("Invalid WAV data layout");
    std::vector<float> pcm(length / width);
    for (size_t i = 0; i < pcm.size(); ++i) {
        const auto *p = bytes.data() + offset + i * width;
        if (format == 1) {
            const std::uint16_t value = u16(p);
            const int signed_value = value < 32768 ? value : static_cast<int>(value) - 65536;
            pcm[i] = static_cast<float>(signed_value) / 32768;
        } else {
            const auto value = u32(p); std::memcpy(&pcm[i], &value, sizeof(float));
        }
    }
    return pcm;
}
} // namespace
int main(int argc, char **argv) {
    CLI::App app{"Native CLAP embeddings (separate from the legacy dummy clap CLI/server)"};
    std::string path, text, audio;
    int threads = 0;
    size_t crop = 0;
    app.add_option("-m,--model", path, "Converted CLAP GGUF")->required()->check(CLI::ExistingFile);
    auto *text_option = app.add_option("-p,--text", text, "UTF-8 text to embed");
    auto *audio_option = app.add_option("-a,--audio", audio, "Mono 48 kHz PCM16/float32 WAV")->check(CLI::ExistingFile);
    text_option->excludes(audio_option);
    app.add_option("-t,--threads", threads, "ggml CPU threads (0: automatic)")->check(CLI::NonNegativeNumber);
    app.add_option("--crop-offset", crop, "Sample offset into audio longer than ten seconds");
    CLI11_PARSE(app, argc, argv);
    try {
        if (text_option->count() + audio_option->count() != 1)
            throw std::runtime_error("Specify exactly one of --text or --audio");
        if (crop && text_option->count()) throw std::runtime_error("--crop-offset requires --audio");
        // Validate/decode audio before paging model weights into memory.
        std::vector<float> pcm;
        if (audio_option->count()) pcm = read_wav(audio);
        clap::Model model(path, threads);
        const auto embedding = text_option->count() ? model.text(text) : model.audio(pcm, 48000, crop);
        std::cout << nlohmann::json(embedding).dump() << '\n';
        if (!std::cout) throw std::runtime_error("Failed to write embedding");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
