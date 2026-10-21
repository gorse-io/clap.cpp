#include "clap.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

static void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: clap-audio-test FIXTURE_DIR");
        const std::filesystem::path dir(argv[1]);
        for (const auto &name : {"mono.wav", "mono.flac", "mono.mp3", "stereo.wav",
                                 "stereo.flac", "stereo.mp3", "unknown-length.flac", "mono.ogg", "stereo.ogg", "pcm24.wav", "float.wav"}) {
            const auto audio = clap::decode_audio((dir / name).string());
            require(audio.sample_rate == 48000, std::string(name) + " sample rate");
            require(audio.samples.size() == 4800, std::string(name) + " frame count");
            const bool stereo = std::string(name).find("stereo") == 0;
            double error = 0;
            for (size_t i = 0; i < audio.samples.size(); ++i) {
                require(std::isfinite(audio.samples[i]), std::string(name) + " non-finite sample");
                const double left = 12000.0 / 32768 * std::sin(2 * 3.141592653589793 * 440 * i / 48000);
                const double right = 12000.0 / 32768 * std::sin(2 * 3.141592653589793 * 880 * i / 48000);
                const double expected = stereo ? (left + right) / 2 : left;
                error += std::pow(audio.samples[i] - expected, 2);
            }
            const double rmse = std::sqrt(error / audio.samples.size());
            const double tolerance = (std::string(name).find(".mp3") != std::string::npos ||
                                      std::string(name).find(".ogg") != std::string::npos) ? 0.02 : 0.00002;
            require(rmse < tolerance, std::string(name) + " waveform mismatch: " + std::to_string(rmse));
            std::cout << name << " RMSE=" << rmse << '\n';
        }
        const auto other_rate = clap::decode_audio((dir / "44100.flac").string());
        require(other_rate.sample_rate == 44100 && other_rate.samples.size() == 4410,
                "decoding must preserve the sample rate");
        const auto vorbis_rate = clap::decode_audio((dir / "44100.ogg").string());
        require(vorbis_rate.sample_rate == 44100 && vorbis_rate.samples.size() == 4410,
                "Vorbis decoding must preserve the sample rate");
        require(clap::decode_audio((dir / "vorbis.dat").string()).samples.size() == 4800,
                "content-based Vorbis detection");
        // Format detection must use contents, not filename extensions.
        const auto decoded = clap::decode_audio((dir / "audio.dat").string());
        require(decoded.samples.size() == 4800, "content-based format detection");
        const auto chained = clap::decode_audio((dir / "chained.ogg").string());
        require(chained.sample_rate == 48000 && chained.samples.size() == 9600,
                "decode every link in a chained Vorbis stream");
        const auto first = clap::decode_audio((dir / "mono.ogg").string());
        const auto second = clap::decode_audio((dir / "second.ogg").string());
        require(second.samples.size() == first.samples.size(), "second Vorbis link frame count");
        for (size_t i = 0; i < first.samples.size(); ++i) {
            require(std::abs(chained.samples[i] - first.samples[i]) < 1e-6f, "first chained Vorbis link PCM");
            require(std::abs(chained.samples[first.samples.size() + i] - second.samples[i]) < 1e-6f,
                    "second chained Vorbis link PCM");
        }
        const auto surround = clap::decode_audio((dir / "surround.wav").string());
        require(surround.samples.size() == 4800, "surround frame count");
        for (float value : surround.samples)
            require(std::abs(value - 10500.0f / 32768) < 1e-6f, "equal-weight surround downmix");
        for (const auto &path : {dir / "truncated.wav", dir / "truncated.flac", dir / "nonfinite.wav",
                                 dir / "missing.wav", dir / "invalid.bin", dir / "empty.wav",
                                 dir / "changed-channels.ogg", dir / "changed-rate.ogg",
                                 dir / "truncated.ogg"}) {
            bool rejected = false;
            try { clap::decode_audio(path.string()); } catch (const std::exception &) { rejected = true; }
            require(rejected, "must reject " + path.string());
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
