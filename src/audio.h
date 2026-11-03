#pragma once

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace clap {

// Internal audio preprocessing helpers; not part of the public model API.

// In-place, unnormalized forward FFT with double-precision intermediates.
// The input length must be a nonzero power of two; otherwise throws invalid_argument.
void fft(std::vector<std::complex<double>> &values);

struct Audio {
    std::vector<float> samples; // Mono float PCM; channels are averaged when decoding.
    int sample_rate = 0;
};

// Decode WAV, MP3 or FLAC by contents. Preserves the source sample rate
// (no resampling); throws on unsupported, empty or invalid audio.
Audio decode_audio(const std::string &path);
// Decode encoded WAV/MP3/FLAC bytes directly; no temporary file is created.
Audio decode_audio(const void *data, std::size_t size);

// Mono linear resampling with an eighth-order anti-alias low-pass filter.
// Output length is ceil(input length * target rate / source rate). The causal
// filter adds startup latency; same-rate input is unchanged. Both rates must be
// positive; empty/non-finite or >1 GiB PCM is rejected.
Audio resample_audio(const Audio &audio, int sample_rate = 48000);

// 48 kHz mono float PCM -> [1001, 64] Slaney log-power mel features.
// HF repeatpad: repeat whole short clips, then zero-pad the remainder.
// Long clips use an explicit crop offset (default: first 10 s), making inference
// reproducible rather than depending on NumPy's random state.
std::vector<float> log_mel(const std::vector<float> &pcm, int sample_rate = 48000, std::size_t crop_offset = 0);

} // namespace clap
