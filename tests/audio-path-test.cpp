#include "audio.h"
#include "c-api-helpers.h"
#include "clap.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace c_api_test;

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 4, "expected model, audio fixture directory and long audio file");
        clap_runtime_options options = clap_default_options();
        options.threads = 1;
        options.device = "cpu";
        CLAP *raw = nullptr;
        check(clap_create(argv[1], &options, &raw));
        Model model(raw, clap_destroy);
        const std::filesystem::path directory(argv[2]);
        for (const auto *name : {"mono.wav", "stereo.wav", "mono.mp3", "mono.flac", "44100.flac", "audio.dat"}) {
            const auto path = (directory / name).string();
            const auto decoded = clap::decode_audio(path);
            const auto expected = take_buffer<float>([&](float **out, size_t *count) {
                const std::vector<float> &api_pcm = decoded.samples;
                return clap_encode_audio(model.get(), api_pcm.data(), api_pcm.size(), decoded.sample_rate, 0, out,
                                         count);
            });
            const auto actual = take_buffer<float>([&](float **out, size_t *count) {
                const std::string &api_path = path;
                return clap_encode_audio_file(model.get(), api_path.c_str(), 0, out, count);
            });
            require(actual.size() == expected.size() && actual.size() == 5, "unexpected projection dimension");
            for (std::size_t i = 0; i < actual.size(); ++i)
                require(std::isfinite(actual[i]) && actual[i] == expected[i], "file and PCM embeddings differ");
        }
        const auto long_audio = clap::decode_audio(argv[3]);
        require(take_buffer<float>([&](float **out, size_t *count) {
                    const std::string &api_path = argv[3];
                    return clap_encode_audio_file(model.get(), api_path.c_str(), 12345, out, count);
                }) == take_buffer<float>([&](float **out, size_t *count) {
                    const std::vector<float> &api_pcm = long_audio.samples;
                    return clap_encode_audio(model.get(), api_pcm.data(), api_pcm.size(), long_audio.sample_rate, 12345,
                                             out, count);
                }),
                "crop offset was not forwarded");
        for (const auto *name : {"missing.wav", "invalid.bin", "empty.wav", "nonfinite.wav"}) {
            bool rejected = false;
            try {
                take_buffer<float>([&](float **out, size_t *count) {
                    const std::string &api_path = (directory / name).string();
                    return clap_encode_audio_file(model.get(), api_path.c_str(), 0, out, count);
                });
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected, "invalid audio file was accepted");
        }
        std::cout << "audio path encoding matches PCM encoding\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
