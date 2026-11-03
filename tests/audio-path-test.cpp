#include "clap.h"
#include "audio.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 4, "expected model, audio fixture directory and long audio file");
        clap::CLAP model(argv[1], clap::RuntimeOptions{1, "cpu"});
        const std::filesystem::path directory(argv[2]);
        for (const auto *name : {"mono.wav", "stereo.wav", "mono.mp3", "mono.flac", "44100.flac", "audio.dat"}) {
            const auto path = (directory / name).string();
            const auto decoded = clap::decode_audio(path);
            const auto expected = model.encode_audio(decoded.samples, decoded.sample_rate);
            const auto actual = model.encode_audio(path);
            require(actual.size() == expected.size() && actual.size() == 5, "unexpected projection dimension");
            for (std::size_t i = 0; i < actual.size(); ++i)
                require(std::isfinite(actual[i]) && actual[i] == expected[i], "file and PCM embeddings differ");
        }
        const auto long_audio = clap::decode_audio(argv[3]);
        require(model.encode_audio(argv[3], 12345) ==
                    model.encode_audio(long_audio.samples, long_audio.sample_rate, 12345),
                "crop offset was not forwarded");
        for (const auto *name : {"missing.wav", "invalid.bin", "empty.wav", "nonfinite.wav"}) {
            bool rejected = false;
            try {
                model.encode_audio((directory / name).string());
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
