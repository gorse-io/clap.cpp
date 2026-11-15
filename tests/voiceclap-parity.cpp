/* Copyright 2026 clap.cpp Authors. SPDX-License-Identifier: Apache-2.0 */
// Optional official-oracle driver. All embedding inference uses the public C API.
// Private access is limited to tokenizer IDs and the waveform frontend.
#include "clap.h"
#include "models/voiceclap.h"
#include "vocab.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace clap::detail {
struct FeatureTestAccess {
    static HostMatrix voiceclap_frontend(const std::vector<float> &pcm) { return VoiceClapModel::frontend(pcm); }
};
} // namespace clap::detail

namespace {
std::vector<float> read_pcm(const char *path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error("cannot open PCM fixture");
    const auto bytes = stream.tellg();
    if (bytes < 0 || static_cast<size_t>(bytes) % sizeof(float))
        throw std::runtime_error("PCM fixture must contain whole float32 samples");
    std::vector<float> result(static_cast<size_t>(bytes) / sizeof(float));
    stream.seekg(0);
    if (!result.empty() && !stream.read(reinterpret_cast<char *>(result.data()), bytes))
        throw std::runtime_error("cannot read PCM fixture");
    return result;
}
void check(clap_status status) {
    if (status != CLAP_OK)
        throw std::runtime_error(clap_last_error());
}
template <typename T> void array(const std::vector<T> &values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            std::cout << ',';
        if (!std::isfinite(static_cast<double>(values[i])))
            throw std::runtime_error("non-finite native result");
        std::cout << values[i];
    }
    std::cout << ']';
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 3)
            throw std::runtime_error("usage: voiceclap-parity MODEL MODE [TEXT|PCM]; tokenize uses TOKENIZER.json");
        std::cout << std::setprecision(9);
        const std::string mode = argv[2];
        if (mode == "tokenize") {
            if (argc != 4)
                throw std::runtime_error("tokenize requires UTF-8 text");
            std::ifstream stream(argv[1]);
            nlohmann::json tokenizer;
            stream >> tokenizer;
            clap::detail::VoiceClapVocabulary vocabulary(tokenizer, 2048);
            std::cout << "{\"ids\":";
            array(vocabulary.tokenize(argv[3], 2048));
            std::cout << "}\n";
            return 0;
        }
        if (mode == "frontend") {
            if (argc != 4)
                throw std::runtime_error("frontend requires PCM fixture");
            auto mel = clap::detail::FeatureTestAccess::voiceclap_frontend(read_pcm(argv[3]));
            std::cout << "{\"frames\":" << mel.rows << ",\"dimension\":" << mel.cols << ",\"values\":";
            array(mel.data);
            std::cout << "}\n";
            return 0;
        }
        clap_runtime_options options = clap_default_options();
        options.device = "cpu";
        options.threads = 1;
        options.activation_precision = CLAP_ACTIVATION_F32;
        CLAP *raw = nullptr;
        const auto loaded = clap_create(argv[1], &options, &raw);
        std::unique_ptr<CLAP, decltype(&clap_destroy)> model(raw, clap_destroy);
        if (mode == "reject") {
            std::cout << "{\"load_status\":" << int(loaded) << "}\n";
            return 0;
        }
        check(loaded);
        const int dimension = clap_projection_dimension(model.get());
        if (dimension <= 0)
            throw std::runtime_error("invalid projection dimension");
        std::vector<float> output(static_cast<size_t>(dimension));
        if (mode == "text") {
            if (argc != 4)
                throw std::runtime_error("text requires UTF-8 text");
            check(clap_encode_text(model.get(), argv[3], output.data(), output.size()));
        } else if (mode == "audio" || mode == "invalid") {
            if (argc != 4)
                throw std::runtime_error("audio/invalid requires PCM fixture");
            auto pcm = read_pcm(argv[3]);
            if (mode == "invalid") {
                const auto empty = clap_encode_audio(model.get(), nullptr, 0, 16000, 0, output.data(), output.size());
                const auto crop =
                    clap_encode_audio(model.get(), pcm.data(), pcm.size(), 16000, 1, output.data(), output.size());
                if (pcm.empty())
                    throw std::runtime_error("invalid-input test needs nonempty control PCM");
                pcm[0] = std::numeric_limits<float>::quiet_NaN();
                const auto nan =
                    clap_encode_audio(model.get(), pcm.data(), pcm.size(), 16000, 0, output.data(), output.size());
                std::cout << "{\"empty\":" << int(empty) << ",\"crop\":" << int(crop) << ",\"nan\":" << int(nan)
                          << "}\n";
                return 0;
            }
            check(clap_encode_audio(model.get(), pcm.data(), pcm.size(), 16000, 0, output.data(), output.size()));
        } else {
            throw std::runtime_error("unknown mode: " + mode);
        }
        std::cout << "{\"values\":";
        array(output);
        std::cout << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "voiceclap-parity: " << error.what() << '\n';
        return 1;
    }
}
