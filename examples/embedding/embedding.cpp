#include "clap.h"
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
#include <utility>

int main(int argc, char **argv) {
    CLI::App app{"Native CLAP/CLaMP3 embeddings (separate from the legacy dummy clap CLI/server)"};
    std::string path, text, audio;
    int threads = 0;
    size_t crop = 0;
    bool frames = false, clamp3_features = false;
    app.add_option("-m,--model", path, "Converted CLAP/CLaMP3/MERT-v1/MERT2/MuQ/MuLan GGUF")->required()->check(CLI::ExistingFile);
    auto *text_option = app.add_option("-p,--text", text, "UTF-8 text to embed");
    auto *audio_option = app.add_option("-a,--audio", audio, "WAV, MP3 or FLAC (mixed to mono and resampled for the model)")->check(CLI::ExistingFile);
    text_option->excludes(audio_option);
    app.add_option("-t,--threads", threads, "ggml CPU threads (0: automatic)")->check(CLI::NonNegativeNumber);
    auto *crop_option = app.add_option("--crop-offset", crop, "Sample offset at 48 kHz into CLAP audio longer than ten seconds");
    app.add_flag("--frames", frames, "Return final-layer waveform-encoder frames instead of mean pooling");
    app.add_flag("--clamp3-features", clamp3_features,
                 "Return standalone MERT-v1 mean-layer/time features (one vector per 5 s window)");
    CLI11_PARSE(app, argc, argv);
    try {
        if (text_option->count() + audio_option->count() != 1)
            throw std::runtime_error("Specify exactly one of --text or --audio");
        if (crop_option->count() && !audio_option->count()) throw std::runtime_error("--crop-offset requires --audio");
        if (frames && (!audio_option->count() || crop_option->count()))
            throw std::runtime_error("--frames requires --audio and excludes --crop-offset");
        if (clamp3_features && (!audio_option->count() || frames || crop_option->count()))
            throw std::runtime_error("--clamp3-features requires --audio and excludes --frames/--crop-offset");
        std::vector<float> pcm;
        int pcm_rate = 0;
        if (audio_option->count()) {
            auto decoded = clap::decode_audio(audio);
            pcm_rate = decoded.sample_rate;
            pcm = std::move(decoded.samples);
        }
        clap::Model model(path, threads);
        if (crop_option->count() && model.audio_sample_rate() != 48000)
            throw std::runtime_error("--crop-offset requires a CLAP model");
        if (frames || clamp3_features) {
            const auto sequence = clamp3_features ? model.clamp3_features(pcm, pcm_rate) : model.audio_frames(pcm, pcm_rate);
            std::cout << nlohmann::json{{"frames", sequence.frames}, {"dimension", sequence.dimension},
                                       {"values", sequence.values}}.dump() << '\n';
        } else {
            const auto embedding = text_option->count() ? model.text(text) : model.audio(pcm, pcm_rate, crop);
            std::cout << nlohmann::json(embedding).dump() << '\n';
        }
        if (!std::cout) throw std::runtime_error("Failed to write representations");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
