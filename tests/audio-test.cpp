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
                                 "stereo.flac", "stereo.mp3", "unknown-length.flac", "pcm24.wav", "float.wav"}) {
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
            const double tolerance = std::string(name).find(".mp3") != std::string::npos ? 0.02 : 0.00002;
            require(rmse < tolerance, std::string(name) + " waveform mismatch: " + std::to_string(rmse));
            std::cout << name << " RMSE=" << rmse << '\n';
        }
        const auto other_rate = clap::decode_audio((dir / "44100.flac").string());
        require(other_rate.sample_rate == 44100 && other_rate.samples.size() == 4410,
                "decoding must preserve the sample rate");
        // Format detection must use contents, not filename extensions.
        const auto decoded = clap::decode_audio((dir / "audio.dat").string());
        require(decoded.samples.size() == 4800, "content-based format detection");
        const auto surround = clap::decode_audio((dir / "surround.wav").string());
        require(surround.samples.size() == 4800, "surround frame count");
        for (float value : surround.samples)
            require(std::abs(value - 10500.0f / 32768) < 1e-6f, "equal-weight surround downmix");
        for (const auto &path : {dir / "truncated.wav", dir / "truncated.flac", dir / "nonfinite.wav",
                                 dir / "missing.wav", dir / "invalid.bin", dir / "empty.wav"}) {
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
