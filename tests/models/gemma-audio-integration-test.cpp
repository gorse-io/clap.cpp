#include "clap.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char **argv) {
    if (argc != 4) return 1;
    try {
        nlohmann::json fixture; std::ifstream(argv[3]) >> fixture;
        const auto pcm = fixture.at("pcm").get<std::vector<float>>();
        const auto expected = fixture.at("embedding").get<std::vector<float>>();
        clap::CLAP model(argv[1], argv[2], 2);
        require(model.audio_sample_rate() == 16000, "wrong audio sample rate");
        const auto actual = model.encode_audio(pcm, 16000);
        require(actual.size() == expected.size(), "wrong audio embedding dimension");
        float maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]), "non-finite audio embedding");
            maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
        }
        std::cout << "public audio scalar oracle max_abs=" << maximum << '\n';
        require(maximum < 5e-5f, "public audio scalar oracle mismatch");
        require(actual == model.encode_audio(pcm, 16000), "audio output changed across calls");
        const auto resampled = clap::resample_audio({pcm, 32000}, 16000);
        require(model.encode_audio(pcm, 32000) == model.encode_audio(resampled.samples, 16000), "automatic resampling differs");
        bool rejected = false;
        try { model.encode_audio(pcm, 16000, 1); } catch (const std::exception &) { rejected = true; }
        require(rejected, "crop offset accepted");
        rejected = false;
        try { model.audio_frames(pcm, 16000); } catch (const std::exception &) { rejected = true; }
        require(rejected, "frame API unexpectedly accepted");
        rejected = false;
        try { model.encode_audio(pcm, 0); } catch (const std::exception &) { rejected = true; }
        require(rejected, "zero sample rate accepted");
        for (const auto &invalid : {std::vector<float>{}, std::vector<float>(160, 0),
                std::vector<float>(320, std::numeric_limits<float>::quiet_NaN())}) {
            rejected = false;
            try { model.encode_audio(invalid, 16000); } catch (const std::exception &) { rejected = true; }
            require(rejected, "invalid waveform accepted");
        }
        clap::CLAP text_only(argv[1], 2);
        rejected = false;
        try { text_only.encode_audio(pcm, 16000); }
        catch (const std::exception &e) { rejected = std::string(e.what()).find("mmproj") != std::string::npos; }
        require(rejected, "missing mmproj was not explained");
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
    return 0;
}
