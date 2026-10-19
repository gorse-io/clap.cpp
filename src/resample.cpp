#include "clap.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace clap {
Audio resample_audio(const Audio &audio, int sample_rate) {
    constexpr size_t max_frames = (size_t{1} << 30) / sizeof(float);
    if (audio.sample_rate <= 0 || sample_rate <= 0)
        throw std::runtime_error("Audio sample rates must be positive");
    if (audio.samples.empty() || audio.samples.size() > max_frames)
        throw std::runtime_error("Audio must be non-empty and at most 1 GiB");
    if (!std::all_of(audio.samples.begin(), audio.samples.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("Non-finite audio sample");
    if (audio.sample_rate == sample_rate) return audio;

    // Round duration up to the next output sample, using bounded integer math.
    const auto frames = (static_cast<std::uint64_t>(audio.samples.size()) * sample_rate +
                         audio.sample_rate - 1) / audio.sample_rate;
    if (frames > max_frames) throw std::runtime_error("Resampled audio exceeds 1 GiB");
    Audio output{std::vector<float>(static_cast<size_t>(frames)), sample_rate};

    // Hann-windowed sinc with 32 lobes on either side. Widen the input-domain
    // filter when downsampling so frequencies above the new Nyquist are removed.
    // A 5% transition band keeps the filter away from the Nyquist discontinuity.
    constexpr double pi = 3.14159265358979323846;
    const double ratio = static_cast<double>(sample_rate) / audio.sample_rate;
    const double cutoff = 0.95 * std::min(1.0, ratio);
    const double radius = 32 / cutoff;
    for (size_t i = 0; i < output.samples.size(); ++i) {
        const double position = static_cast<double>(i) * audio.sample_rate / sample_rate;
        const auto first = static_cast<size_t>(std::ceil(std::max(0.0, position - radius)));
        const auto last = static_cast<size_t>(std::floor(std::min(
            static_cast<double>(audio.samples.size() - 1), position + radius)));
        double sum = 0, weights = 0;
        for (size_t j = first; j <= last; ++j) {
            const double distance = position - static_cast<double>(j);
            const double x = pi * cutoff * distance;
            const double sinc = std::abs(x) < 1e-12 ? 1.0 : std::sin(x) / x;
            const double window = 0.5 * (1 + std::cos(pi * distance / radius));
            const double weight = cutoff * sinc * window;
            sum += weight * audio.samples[j];
            weights += weight;
        }
        // Truncate and renormalize at boundaries: constant clips keep unity gain
        // rather than fading toward zero. The centered filter adds no delay.
        output.samples[i] = static_cast<float>(sum / weights);
        if (!std::isfinite(output.samples[i]))
            throw std::runtime_error("Non-finite resampled audio sample");
    }
    return output;
}
} // namespace clap
