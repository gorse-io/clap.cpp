#include "clap.h"

#include "miniaudio-config.h"
#include "minivorbis.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace clap {
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
    // miniaudio has no built-in Vorbis decoder. Retain MiniVorbis for its
    // seekable, homogeneous chained-stream support (stb_vorbis lacks chaining).
    OggVorbis_File vorbis{};
    if (ov_fopen(path.c_str(), &vorbis) == 0) {
        const auto close = [](OggVorbis_File *decoder) { ov_clear(decoder); };
        const std::unique_ptr<OggVorbis_File, decltype(close)> guard(&vorbis, close);
        const auto *info = ov_info(&vorbis, 0);
        if (!info || info->channels <= 0 || info->rate <= 0 ||
            info->rate > std::numeric_limits<int>::max())
            throw std::runtime_error("Invalid Ogg Vorbis channel count or sample rate: " + path);
        const unsigned channels = static_cast<unsigned>(info->channels);
        const unsigned sample_rate = static_cast<unsigned>(info->rate);
        // Audio has a single sample rate; do not silently misinterpret chained links.
        for (long link = 1; link < ov_streams(&vorbis); ++link) {
            const auto *next = ov_info(&vorbis, static_cast<int>(link));
            if (!next || next->channels != info->channels || next->rate != info->rate)
                throw std::runtime_error("Ogg Vorbis links change audio format: " + path);
        }
        const auto total = ov_pcm_total(&vorbis, -1);
        if (ov_pcm_seek(&vorbis, 0) != 0)
            throw std::runtime_error("Cannot seek Ogg Vorbis audio: " + path);
        auto audio = read_mono(channels, sample_rate, [&](size_t count, float *out) -> size_t {
            float **pcm = nullptr;
            int section = 0;
            const long frames = ov_read_float(&vorbis, &pcm, static_cast<int>(count), &section);
            if (frames < 0) throw std::runtime_error("Invalid Ogg Vorbis audio: " + path);
            for (long i = 0; i < frames; ++i)
                for (unsigned c = 0; c < channels; ++c)
                    out[static_cast<size_t>(i) * channels + c] = pcm[c][i];
            return static_cast<size_t>(frames);
        });
        if (total < 0 || audio.samples.size() != static_cast<size_t>(total))
            throw std::runtime_error("Truncated Ogg Vorbis audio: " + path);
        return audio;
    }
    throw std::runtime_error("Cannot decode audio (expected WAV, MP3, FLAC or Ogg Vorbis): " + path);
}
} // namespace clap
