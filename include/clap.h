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

// Time-major frame representations returned by waveform encoders such as
// MERT-v1-95M/MERT-v2-FullSong. values has exactly frames * dimension elements.
struct AudioFrames {
    std::vector<float> values;
    int frames = 0;
    int dimension = 0;
};

// Decode WAV, MP3 or FLAC by contents. Preserves the source sample rate
// (no resampling); throws on unsupported, empty or invalid audio.
Audio decode_audio(const std::string &path);

// Mono linear resampling with an eighth-order anti-alias low-pass filter.
// Output length is ceil(input length * target rate / source rate). The causal
// filter adds startup latency; same-rate input is unchanged. Both rates must be
// positive; empty/non-finite or >1 GiB PCM is rejected.
Audio resample_audio(const Audio &audio, int sample_rate = 48000);

// 48 kHz mono float PCM -> [1001, 64] Slaney log-power mel features.
// HF repeatpad: repeat whole short clips, then zero-pad the remainder.
// Long clips use an explicit crop offset (default: first 10 s), making inference
// reproducible rather than depending on NumPy's random state.
std::vector<float> log_mel(const std::vector<float> &pcm, int sample_rate = 48000,
                           std::size_t crop_offset = 0);

// Native CLAP, CLaMP3, MERT-v1, MERT2 or MuQ, selected from GGUF metadata. Weights are
// memory-mapped on POSIX platforms; MuQ F16 weights expand to owned F32 buffers.
// Each call embeds one item; methods on the same instance are not thread-safe.
// CLAP returns unit-normalized projections. CLaMP3 returns its native unnormalized
// pooled projections from MERT-v1-95M features. Chain a separate MERT Model's
// clamp3_features() into audio_features() for native raw-audio inference.
// MERT-v1 audio_frames() exposes final-layer frames; audio() mean-pools them.
// Neither is the all-layer CLaMP3 representation: use clamp3_features() for that.
// MERT-v2-FullSong and MuQ audio() mean-pool the complete input's final frames;
// audio_frames() returns the unpooled sequence (approximately 25 Hz for published weights).
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
    // CLAP: [frames, audio_config.num_mel_bins], channel count = 1.
    // CLaMP3: time-major [windows, audio_feature_dim] MERT features (published: 768).
    // MERT waveform encoders do not accept precomputed features.
    std::vector<float> audio_features(const std::vector<float> &features, int frames);
    std::vector<float> audio(const std::vector<float> &pcm, int sample_rate = 48000,
                              std::size_t crop_offset = 0);
    AudioFrames audio_frames(const std::vector<float> &pcm, int sample_rate = 48000);
    // MERT-v1 CLaMP3 --mean_features preprocessing: normalize the entire mono
    // recording, split into non-overlapping 5 s windows, discard a final window
    // shorter than 1 s, and average time then all hidden states (embedding +
    // transformer layers). Returns one feature vector per retained window.
    // At least one second is required. Resampling uses the native audio helper.
    AudioFrames clamp3_features(const std::vector<float> &pcm, int sample_rate = 24000);
    int audio_sample_rate() const;
    int projection_dimension() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace clap
