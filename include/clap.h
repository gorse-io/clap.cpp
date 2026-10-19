#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace clap {

struct Audio {
    std::vector<float> samples; // Mono float PCM; channels are averaged when decoding.
    int sample_rate = 0;
};

// Decode WAV, MP3 or FLAC by file contents. Preserves the source sample rate
// (no resampling); throws on unsupported, empty or invalid audio.
Audio decode_audio(const std::string &path);

// 48 kHz mono float PCM -> [1001, 64] Slaney log-power mel features.
// HF repeatpad: repeat whole short clips, then zero-pad the remainder.
// Long clips use an explicit crop offset (default: first 10 s), making inference
// reproducible rather than depending on NumPy's random state.
std::vector<float> log_mel(const std::vector<float> &pcm, int sample_rate = 48000,
                           std::size_t crop_offset = 0);

// Unfused HTSAT + RoBERTa CLAP. Weights are memory-mapped on POSIX platforms.
// Each call embeds one item; methods on the same instance are not thread-safe.
// Returns unit-normalized projection vectors, as HF get_*_features does.
class Model {
public:
    explicit Model(const std::string &gguf_path, int threads = 0);
    ~Model();
    Model(Model &&) noexcept;
    Model &operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    std::vector<int> tokenize(const std::string &utf8, std::size_t max_length = 512) const;
    std::vector<float> text(const std::vector<int> &token_ids);
    std::vector<float> text(const std::string &utf8);
    // Feature layout [frames, audio_config.num_mel_bins], channel count = 1.
    std::vector<float> audio_features(const std::vector<float> &features, int frames);
    std::vector<float> audio(const std::vector<float> &pcm, int sample_rate = 48000,
                              std::size_t crop_offset = 0);
    int projection_dimension() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace clap
