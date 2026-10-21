#include "clap.h"
#include "miniaudio-config.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <memory>

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

    auto config = ma_resampler_config_init(ma_format_f32, 1,
        static_cast<ma_uint32>(audio.sample_rate), static_cast<ma_uint32>(sample_rate),
        ma_resample_algorithm_linear);
    // Use miniaudio's highest built-in LPF order to suppress downsampling aliases.
    config.linear.lpfOrder = MA_MAX_FILTER_ORDER;
    ma_resampler resampler{};
    if (ma_resampler_init(&config, nullptr, &resampler) != MA_SUCCESS)
        throw std::runtime_error("Cannot initialize audio resampler");
    const auto close = [](ma_resampler *value) { ma_resampler_uninit(value, nullptr); };
    const std::unique_ptr<ma_resampler, decltype(close)> guard(&resampler, close);

    // This is a causal linear + LPF resampler, not the previous zero-phase sinc.
    // Keep its startup latency, but preserve ceil(duration * target rate). If
    // the last fractional frame needs lookahead, extend the final input value
    // rather than inserting silence or returning a short output.
    size_t consumed = 0, written = 0;
    const std::vector<float> tail(4096, audio.samples.back());
    while (written < output.samples.size()) {
        const bool flushing = consumed == audio.samples.size();
        ma_uint64 input_frames = flushing ? tail.size() : audio.samples.size() - consumed;
        ma_uint64 output_frames = output.samples.size() - written;
        const float *input = flushing ? tail.data() : audio.samples.data() + consumed;
        if (ma_resampler_process_pcm_frames(&resampler, input, &input_frames,
                output.samples.data() + written, &output_frames) != MA_SUCCESS)
            throw std::runtime_error("Cannot resample audio");
        if (!input_frames && !output_frames) throw std::runtime_error("Audio resampler made no progress");
        if (!flushing) consumed += static_cast<size_t>(input_frames);
        written += static_cast<size_t>(output_frames);
    }
    if (!std::all_of(output.samples.begin(), output.samples.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("Non-finite resampled audio sample");
    return output;
}
} // namespace clap
