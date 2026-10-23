#include "clap.h"
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(clap_status status) {
    if (status != CLAP_STATUS_SUCCESS) throw std::runtime_error(clap_last_error());
}
struct Model {
    clap_model *value = nullptr;
    ~Model() { clap_model_free(value); }
};
struct Audio {
    clap_audio value{};
    ~Audio() { clap_audio_free(&value); }
};
struct Buffer {
    clap_float_buffer value{};
    ~Buffer() { clap_float_buffer_free(&value); }
};
struct Frames {
    clap_audio_frames value{};
    ~Frames() { clap_audio_frames_free(&value); }
};
} // namespace

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
        Audio decoded;
        if (audio_option->count()) check(clap_audio_decode(audio.c_str(), &decoded.value));
        Model model;
        auto params = clap_model_default_params();
        params.n_threads = threads;
        check(clap_model_load(path.c_str(), params, &model.value));
        int32_t rate = 0;
        check(clap_model_audio_sample_rate(model.value, &rate));
        if (crop_option->count() && rate != 48000)
            throw std::runtime_error("--crop-offset requires a CLAP model");
        if (frames || clamp3_features) {
            Frames sequence;
            check(clamp3_features ?
                  clap_model_clamp3_features(model.value, decoded.value.samples, decoded.value.n_samples,
                                             decoded.value.sample_rate, &sequence.value) :
                  clap_model_audio_frames(model.value, decoded.value.samples, decoded.value.n_samples,
                                          decoded.value.sample_rate, &sequence.value));
            const std::vector<float> values(sequence.value.values, sequence.value.values + sequence.value.n_values);
            std::cout << nlohmann::json{{"frames", sequence.value.frames}, {"dimension", sequence.value.dimension},
                                       {"values", values}}.dump() << '\n';
        } else {
            Buffer embedding;
            auto ap = clap_audio_default_params();
            ap.sample_rate = decoded.value.sample_rate;
            ap.crop_offset = crop;
            check(text_option->count() ? clap_model_text(model.value, text.c_str(), &embedding.value) :
                  clap_model_audio(model.value, decoded.value.samples, decoded.value.n_samples, ap, &embedding.value));
            std::cout << nlohmann::json(std::vector<float>(embedding.value.data,
                                                          embedding.value.data + embedding.value.size)).dump() << '\n';
        }
        if (!std::cout) throw std::runtime_error("Failed to write representations");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
