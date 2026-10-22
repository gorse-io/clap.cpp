#include "clap.h"
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <utility>

namespace {
struct NpyFeatures {
    std::vector<float> values;
    int frames = 0;
};

NpyFeatures load_npy(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open NumPy feature file");
    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (input.gcount() != 8 || std::string(magic, 6) != "\x93NUMPY")
        throw std::runtime_error("Invalid NumPy .npy header");
    const unsigned major = static_cast<unsigned char>(magic[6]);
    uint32_t header_size = 0;
    if (major == 1) {
        unsigned char length[2]{}; input.read(reinterpret_cast<char *>(length), 2);
        header_size = static_cast<uint32_t>(length[0]) | static_cast<uint32_t>(length[1]) << 8;
    } else if (major == 2 || major == 3) {
        unsigned char length[4]{}; input.read(reinterpret_cast<char *>(length), 4);
        header_size = static_cast<uint32_t>(length[0]) | static_cast<uint32_t>(length[1]) << 8 |
                      static_cast<uint32_t>(length[2]) << 16 | static_cast<uint32_t>(length[3]) << 24;
    } else throw std::runtime_error("Unsupported NumPy .npy version");
    if (!input || header_size == 0 || header_size > 1024 * 1024)
        throw std::runtime_error("Invalid NumPy .npy header size");
    std::string header(header_size, '\0'); input.read(header.data(), header.size());
    if (!input) throw std::runtime_error("Truncated NumPy .npy header");
    std::smatch match;
    if (!std::regex_search(header, match, std::regex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]")) ||
        (match[1] != "<f4" && match[1] != "=f4"))
        throw std::runtime_error("CLaMP3 NumPy input must contain little-endian float32 features");
    if (!std::regex_search(header, match, std::regex("['\"]fortran_order['\"]\\s*:\\s*(True|False)")) || match[1] != "False")
        throw std::runtime_error("Fortran-order NumPy arrays are unsupported");
    if (!std::regex_search(header, match, std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)")))
        throw std::runtime_error("NumPy feature shape is missing");
    std::vector<size_t> shape;
    const std::string dimensions = match[1];
    if (!std::regex_match(dimensions, std::regex("\\s*[0-9]+\\s*,\\s*[0-9]+\\s*(,\\s*[0-9]+\\s*)?,?\\s*")))
        throw std::runtime_error("Invalid NumPy feature shape");
    const std::regex number("[0-9]+");
    for (std::sregex_iterator it(dimensions.begin(), dimensions.end(), number), end; it != end; ++it)
        shape.push_back(static_cast<size_t>(std::stoull((*it).str())));
    if (!((shape.size() == 2 && shape[0] > 0 && shape[1] > 0) ||
          (shape.size() == 3 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0)))
        throw std::runtime_error("CLaMP3 NumPy shape must be [frames, features] or [1, frames, features]");
    const size_t frames = shape.size() == 2 ? shape[0] : shape[1];
    const size_t width = shape.back();
    if (frames > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        width > std::numeric_limits<size_t>::max() / frames || frames * width > (1ULL << 30) / sizeof(float))
        throw std::runtime_error("NumPy feature array is too large");
    NpyFeatures result;
    result.frames = static_cast<int>(frames);
    result.values.resize(frames * width);
    input.read(reinterpret_cast<char *>(result.values.data()), static_cast<std::streamsize>(result.values.size() * sizeof(float)));
    if (!input) throw std::runtime_error("Truncated NumPy feature data");
    char extra;
    if (input.read(&extra, 1)) throw std::runtime_error("Unexpected data after NumPy feature array");
    return result;
}
} // namespace

int main(int argc, char **argv) {
    CLI::App app{"Native CLAP/CLaMP3 embeddings (separate from the legacy dummy clap CLI/server)"};
    std::string path, text, audio, audio_features;
    int threads = 0;
    size_t crop = 0;
    bool frames = false;
    app.add_option("-m,--model", path, "Converted CLAP/CLaMP3/MERT2/MuQ GGUF")->required()->check(CLI::ExistingFile);
    auto *text_option = app.add_option("-p,--text", text, "UTF-8 text to embed");
    auto *audio_option = app.add_option("-a,--audio", audio, "WAV, MP3 or FLAC (mixed to mono and resampled for the model)")->check(CLI::ExistingFile);
    auto *features_option = app.add_option("--audio-features", audio_features,
        "CLaMP3 MERT-v1-95M float32 .npy array ([T,768] or [1,T,768])")->check(CLI::ExistingFile);
    text_option->excludes(audio_option);
    text_option->excludes(features_option);
    audio_option->excludes(features_option);
    app.add_option("-t,--threads", threads, "ggml CPU threads (0: automatic)")->check(CLI::NonNegativeNumber);
    app.add_option("--crop-offset", crop, "Sample offset at 48 kHz into audio longer than ten seconds");
    app.add_flag("--frames", frames, "Return waveform-encoder frame representations instead of mean pooling");
    CLI11_PARSE(app, argc, argv);
    try {
        if (text_option->count() + audio_option->count() + features_option->count() != 1)
            throw std::runtime_error("Specify exactly one of --text, --audio or --audio-features");
        if (crop && !audio_option->count()) throw std::runtime_error("--crop-offset requires --audio");
        if (frames && !audio_option->count()) throw std::runtime_error("--frames requires --audio");
        if (frames && crop) throw std::runtime_error("--frames does not support --crop-offset");
        // Validate/decode audio before paging model weights into memory.
        std::vector<float> pcm;
        int pcm_rate = 0;
        if (audio_option->count()) {
            auto decoded = clap::decode_audio(audio);
            pcm_rate = decoded.sample_rate;
            pcm = std::move(decoded.samples);
        }
        NpyFeatures mert;
        if (features_option->count()) mert = load_npy(audio_features);
        clap::Model model(path, threads);
        if (audio_option->count() && model.audio_sample_rate() > 0 && pcm_rate != model.audio_sample_rate()) {
            auto converted = clap::resample_audio({std::move(pcm), pcm_rate}, model.audio_sample_rate());
            pcm = std::move(converted.samples);
            pcm_rate = converted.sample_rate;
        }
        if (frames) {
            const auto sequence = model.audio_frames(pcm, pcm_rate);
            std::cout << nlohmann::json{{"frames", sequence.frames}, {"dimension", sequence.dimension},
                                       {"values", sequence.values}}.dump() << '\n';
            if (!std::cout) throw std::runtime_error("Failed to write frame representations");
            return 0;
        }
        const auto embedding = text_option->count() ? model.text(text) :
                               features_option->count() ? model.audio_features(mert.values, mert.frames) :
                               model.audio(pcm, pcm_rate, crop);
        std::cout << nlohmann::json(embedding).dump() << '\n';
        if (!std::cout) throw std::runtime_error("Failed to write embedding");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
