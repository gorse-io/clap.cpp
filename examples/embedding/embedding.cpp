#include "clap.h"
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
#include <utility>

int main(int argc, char **argv) {
    CLI::App app{"Native CLAP embeddings (separate from the legacy dummy clap CLI/server)"};
    std::string path, text, audio;
    int threads = 0;
    size_t crop = 0;
    app.add_option("-m,--model", path, "Converted CLAP GGUF")->required()->check(CLI::ExistingFile);
    auto *text_option = app.add_option("-p,--text", text, "UTF-8 text to embed");
    auto *audio_option = app.add_option("-a,--audio", audio, "48 kHz WAV, MP3 or FLAC (automatically downmixed to mono)")->check(CLI::ExistingFile);
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
        if (audio_option->count()) {
            auto decoded = clap::decode_audio(audio);
            if (decoded.sample_rate != 48000)
                throw std::runtime_error("Audio must be 48000 Hz; resample before inference");
            pcm = std::move(decoded.samples);
        }
        clap::Model model(path, threads);
        const auto embedding = text_option->count() ? model.text(text) : model.audio(pcm, 48000, crop);
        std::cout << nlohmann::json(embedding).dump() << '\n';
        if (!std::cout) throw std::runtime_error("Failed to write embedding");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
