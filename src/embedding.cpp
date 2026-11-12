#include "audio.h"
#include "clap.h"
#include "embedding.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Vector = std::vector<float>;
using nlohmann::json;

const char *c_string(const std::string &value) {
    if (value.find('\0') != std::string::npos)
        throw std::invalid_argument("text and paths must contain no NUL bytes");
    return value.c_str();
}

void check(clap_status status) {
    if (status == CLAP_OK)
        return;
    const std::string message = clap_last_error();
    if (status == CLAP_INVALID_ARGUMENT)
        throw std::invalid_argument(message);
    if (status == CLAP_OUT_OF_MEMORY)
        throw std::bad_alloc();
    throw std::runtime_error(message);
}

template <typename T> std::vector<T> take_buffer(T *data, std::size_t count) {
    const std::unique_ptr<T, decltype(&clap_free)> owned(data, clap_free);
    if (!count)
        return {};
    return {data, data + count};
}

} // namespace

int cmd_embedding(const EmbeddingOptions &options) {
    try {
        // Preserve native CLI error precedence: decode audio before loading weights.
        clap::Audio decoded;
        if (!options.audio_path.empty())
            decoded = clap::decode_audio(options.audio_path);
        auto runtime = clap_default_options();
        runtime.threads = options.threads;
        runtime.device = c_string(options.device);
        CLAP *handle = nullptr;
        check(clap_create(c_string(options.model_path), &runtime, &handle));
        const std::unique_ptr<CLAP, decltype(&clap_destroy)> model(handle, clap_destroy);
        if (!options.audio_path.empty()) {
            const int sample_rate = clap_audio_sample_rate(model.get());
            if (options.crop_requested && sample_rate != 48000) {
                throw std::runtime_error("--crop-offset requires a CLAP model");
            }
            if (options.frames) {
                clap_audio_frames sequence{};
                check(clap_get_audio_frames(model.get(), decoded.samples.data(), decoded.samples.size(),
                                            decoded.sample_rate, &sequence));
                const std::unique_ptr<clap_audio_frames, decltype(&clap_free_audio_frames)> owned(
                    &sequence, clap_free_audio_frames);
                const Vector values =
                    sequence.frames && sequence.dimension
                        ? Vector(sequence.values, sequence.values + sequence.frames * sequence.dimension)
                        : Vector{};
                std::cout
                    << json{{"frames", sequence.frames}, {"dimension", sequence.dimension}, {"values", values}}.dump()
                    << '\n';
            } else {
                float *values = nullptr;
                std::size_t count = 0;
                check(clap_encode_audio(model.get(), decoded.samples.data(), decoded.samples.size(),
                                        decoded.sample_rate, options.crop_offset, &values, &count));
                std::cout << json(take_buffer(values, count)).dump() << '\n';
            }
        } else {
            float *values = nullptr;
            std::size_t count = 0;
            check(clap_encode_text(model.get(), c_string(options.text), &values, &count));
            std::cout << json(take_buffer(values, count)).dump() << '\n';
        }
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("Failed to write embedding output");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
