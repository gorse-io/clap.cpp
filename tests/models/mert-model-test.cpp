#include "clap.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F function) {
    try { function(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid MERT API input was accepted");
}
void near(const std::vector<float> &a, const std::vector<float> &b, float tolerance) {
    require(a.size() == b.size(), "MERT API output size mismatch");
    for (size_t i = 0; i < a.size(); ++i)
        require(std::isfinite(a[i]) && std::isfinite(b[i]) && std::abs(a[i] - b[i]) <= tolerance,
                "MERT API output mismatch");
}
void u32(std::ofstream &stream, uint32_t value, int bytes = 4) {
    for (int i = 0; i < bytes; ++i) stream.put(static_cast<char>((value >> (8*i)) & 255));
}
void wav(const std::string &path, const clap::Audio &audio) {
    std::ofstream stream(path, std::ios::binary);
    const auto bytes = static_cast<uint32_t>(audio.samples.size() * sizeof(float));
    stream.write("RIFF", 4); u32(stream, 36 + bytes); stream.write("WAVEfmt ", 8);
    u32(stream, 16); u32(stream, 3, 2); u32(stream, 1, 2); u32(stream, audio.sample_rate);
    u32(stream, audio.sample_rate * 4); u32(stream, 4, 2); u32(stream, 32, 2);
    stream.write("data", 4); u32(stream, bytes);
    for (float value : audio.samples) {
        uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); u32(stream, bits);
    }
    require(static_cast<bool>(stream), "cannot write MERT resampling fixture");
}
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: mert-model-test model.gguf output-directory");
        clap::Model model(argv[1], 2);
        require(model.audio_sample_rate() == 24000, "wrong MERT sample rate");
        rejects([&] { model.text("unsupported"); });
        rejects([&] { model.text(std::vector<int>{0}); });
        rejects([&] { model.audio_features(std::vector<float>(16), 1); });
        rejects([&] { model.audio_frames({}); });
        rejects([&] { model.audio_frames(std::vector<float>(399), 24000); });
        rejects([&] { model.clamp3_features(std::vector<float>(23999), 24000); });
        rejects([&] { model.clamp3_features(std::vector<float>(24000), 0); });
        rejects([&] { model.audio_frames(std::vector<float>(24000), -1); });
        rejects([&] { model.audio(std::vector<float>(24000), 24000, 1); });
        auto nonfinite = std::vector<float>(24000);
        nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { model.audio_frames(nonfinite, 24000); });
        rejects([&] { model.clamp3_features(nonfinite, 24000); });

        clap::Audio input; input.sample_rate = 48000; input.samples.resize(48000);
        for (size_t i = 0; i < input.samples.size(); ++i)
            input.samples[i] = 0.2f + 0.1f * std::sin(static_cast<float>(i) * 0.057f) +
                               0.03f * std::cos(static_cast<float>(i) * 0.141f);
        const auto control = clap::resample_audio(input, 24000);
        require(control.samples.size() == 24000, "unexpected resampled MERT duration");
        const auto features = model.clamp3_features(control.samples, 24000);
        require(features.frames == 1 && features.dimension == model.projection_dimension(),
                "wrong MERT feature dimensions");
        near(features.values, model.clamp3_features(input.samples, 48000).values, 0);
        const auto frames = model.audio_frames(control.samples, 24000);
        require(frames.frames == 74 && frames.dimension == model.projection_dimension(), "wrong MERT frame geometry");
        near(frames.values, model.audio_frames(input.samples, 48000).values, 0);
        std::vector<float> mean(frames.dimension);
        for (int row = 0; row < frames.frames; ++row)
            for (int col = 0; col < frames.dimension; ++col)
                mean[col] += frames.values[static_cast<size_t>(row)*frames.dimension+col] / frames.frames;
        near(mean, model.audio(control.samples, 24000), 0);
        wav(std::string(argv[2]) + "/resampled-input.wav", input);
        wav(std::string(argv[2]) + "/resampled-control.wav", control);
        std::cout << "MERT public API: pooling, invalid inputs, native/resampled inference passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
