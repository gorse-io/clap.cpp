#ifndef NOMINMAX
#define NOMINMAX
#endif

#define MINIAUDIO_IMPLEMENTATION
// Decode/convert only: no devices, background workers, engine or encoders.
#define MA_NO_DEVICE_IO
#define MA_NO_ENGINE
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_THREADING
#define MA_NO_RUNTIME_LINKING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#include "miniaudio.h"

#include "clap.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>

namespace clap {
namespace {
constexpr double pi = 3.14159265358979323846;
} // namespace

void fft(std::vector<std::complex<double>> &a) {
    const size_t n = a.size();
    if (n == 0 || (n & (n - 1)) != 0)
        throw std::invalid_argument("FFT length must be a nonzero power of two");
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const auto root = std::polar(1.0, -2 * pi / static_cast<double>(len));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1);
            for (size_t j = 0; j < len / 2; ++j) {
                const auto u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v; a[i + j + len / 2] = u - v; w *= root;
            }
        }
    }
}

namespace {
double hz_to_mel(double hz) {
    return hz < 1000 ? hz / (200.0 / 3) : 15 + std::log(hz / 1000) / (std::log(6.4) / 27);
}
double mel_to_hz(double mel) {
    return mel < 15 ? mel * (200.0 / 3) : 1000 * std::exp((mel - 15) * (std::log(6.4) / 27));
}
} // namespace

std::vector<float> log_mel(const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
    if (sample_rate != 48000)
        throw std::runtime_error("CLAP: audio must be mono 48000 Hz (resample before inference)");
    if (pcm.empty())
        throw std::runtime_error("CLAP: empty waveform");
    if (!(std::all_of(pcm.begin(), pcm.end(), [](float x) { return std::isfinite(x); })))
        throw std::runtime_error("CLAP: non-finite waveform");
    constexpr size_t length = 480000;
    std::vector<double> wav(length, 0);
    if (pcm.size() > length) {
        if (!(crop_offset <= pcm.size() - length))
            throw std::runtime_error("CLAP: crop offset out of range");
        std::copy_n(pcm.data() + crop_offset, length, wav.data());
    } else {
        if (crop_offset != 0)
            throw std::runtime_error("CLAP: crop offset requires audio longer than 10 seconds");
        const size_t repeats = length / pcm.size();
        for (size_t i = 0; i < repeats; ++i) std::copy(pcm.begin(), pcm.end(), wav.begin() + i * pcm.size());
    }
    double filters[64][513];
    double boundaries[66];
    const double low = hz_to_mel(50), high = hz_to_mel(14000);
    for (int i = 0; i < 66; ++i) boundaries[i] = mel_to_hz(low + (high - low) * i / 65);
    for (int m = 0; m < 64; ++m) for (int f = 0; f <= 512; ++f) {
        const double hz = 48000.0 * f / 1024;
        filters[m][f] = std::max(0.0, std::min((hz - boundaries[m]) / (boundaries[m + 1] - boundaries[m]),
                                                   (boundaries[m + 2] - hz) / (boundaries[m + 2] - boundaries[m + 1])))
                         * (2 / (boundaries[m + 2] - boundaries[m]));
    }
    std::vector<float> output(1001 * 64);
    std::vector<std::complex<double>> frame(1024);
    double power[513], window[1024];
    for (int i = 0; i < 1024; ++i) window[i] = 0.5 - 0.5 * std::cos(2 * pi * i / 1024);
    for (int t = 0; t <= 1000; ++t) {
        for (int i = 0; i < 1024; ++i) {
            int pos = t * 480 + i - 512;
            if (pos < 0) pos = -pos;
            if (pos >= static_cast<int>(length)) pos = 2 * static_cast<int>(length) - 2 - pos;
            frame[i] = wav[pos] * window[i];
        }
        fft(frame);
        for (int i = 0; i <= 512; ++i) power[i] = std::norm(frame[i]);
        for (int m = 0; m < 64; ++m) {
            double sum = 0;
            for (int f = 0; f <= 512; ++f) sum += filters[m][f] * power[f];
            output[t * 64 + m] = static_cast<float>(10 * std::log10(std::max(sum, 1e-10)));
        }
    }
    return output;
}

namespace {
// Bound decoded output to 1 GiB, without trusting frame counts in file headers.
constexpr size_t max_frames = (size_t{1} << 30) / sizeof(float);

template <typename Read>
Audio read_mono(unsigned channels, unsigned sample_rate, Read read) {
    if (channels == 0 || channels > 256 || sample_rate == 0 ||
        sample_rate > static_cast<unsigned>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Invalid audio channel count or sample rate");
    Audio audio;
    audio.sample_rate = static_cast<int>(sample_rate);
    // Preserve equal-weight averaging, including LFE in multichannel files.
    const auto config = ma_channel_converter_config_init(
        ma_format_f32, channels, nullptr, 1, nullptr, ma_channel_mix_mode_rectangular);
    ma_channel_converter converter{};
    if (ma_channel_converter_init(&config, nullptr, &converter) != MA_SUCCESS)
        throw std::runtime_error("Cannot initialize mono conversion");
    const auto close = [](ma_channel_converter *value) { ma_channel_converter_uninit(value, nullptr); };
    const std::unique_ptr<ma_channel_converter, decltype(close)> guard(&converter, close);
    constexpr size_t chunk_frames = 4096;
    std::vector<float> chunk(chunk_frames * channels), mono(chunk_frames);
    for (;;) {
        const auto count = read(chunk_frames, chunk.data());
        if (!count) break;
        if (count > chunk_frames || count > max_frames - audio.samples.size())
            throw std::runtime_error("Decoded audio exceeds 1 GiB");
        for (size_t i = 0; i < count * channels; ++i)
            if (!std::isfinite(chunk[i])) throw std::runtime_error("Non-finite audio sample");
        if (ma_channel_converter_process_pcm_frames(&converter, mono.data(), chunk.data(), count) != MA_SUCCESS)
            throw std::runtime_error("Cannot convert audio to mono");
        for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(mono[i])) throw std::runtime_error("Non-finite mono sample");
        audio.samples.insert(audio.samples.end(), mono.begin(), mono.begin() + count);
    }
    if (audio.samples.empty()) throw std::runtime_error("Empty or invalid audio");
    return audio;
}
} // namespace

Audio decode_audio(const std::string &path) {
    // Decode native-rate interleaved float PCM. Keep channels until after the
    // finite-sample check so a bad channel cannot be hidden by downmixing.
    auto config = ma_decoder_config_init(ma_format_f32, 0, 0);
    ma_decoder decoder{};
    if (ma_decoder_init_file(path.c_str(), &config, &decoder) == MA_SUCCESS) {
        const auto close = [](ma_decoder *value) { ma_decoder_uninit(value); };
        const std::unique_ptr<ma_decoder, decltype(close)> guard(&decoder, close);
        ma_uint32 channels = 0, sample_rate = 0;
        if (ma_decoder_get_data_format(&decoder, nullptr, &channels, &sample_rate, nullptr, 0) != MA_SUCCESS)
            throw std::runtime_error("Cannot read audio format: " + path);
        ma_uint64 total = 0;
        // Some FLAC files have an unknown length. MP3 length queries may scan
        // the stream; do not allocate from any untrusted declared frame count.
        const bool known_length = ma_decoder_get_length_in_pcm_frames(&decoder, &total) == MA_SUCCESS && total != 0;
        auto audio = read_mono(channels, sample_rate, [&](size_t count, float *out) {
            ma_uint64 frames = 0;
            const auto result = ma_decoder_read_pcm_frames(&decoder, out, count, &frames);
            if (result != MA_SUCCESS && result != MA_AT_END)
                throw std::runtime_error("Invalid audio: " + path);
            return static_cast<size_t>(frames);
        });
        if (known_length && audio.samples.size() != total)
            throw std::runtime_error("Truncated audio: " + path);
        return audio;
    }
    throw std::runtime_error("Cannot decode audio (expected WAV, MP3 or FLAC): " + path);
}

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
