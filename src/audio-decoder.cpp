#include "clap.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"
#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"
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
    constexpr size_t chunk_frames = 4096;
    std::vector<float> chunk(chunk_frames * channels);
    for (;;) {
        const auto count = read(chunk_frames, chunk.data());
        if (!count) break;
        if (count > max_frames - audio.samples.size())
            throw std::runtime_error("Decoded audio exceeds 1 GiB");
        for (size_t i = 0; i < count; ++i) {
            double sum = 0;
            for (unsigned c = 0; c < channels; ++c) {
                const float value = chunk[i * channels + c];
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite audio sample");
                sum += value;
            }
            audio.samples.push_back(static_cast<float>(sum / channels));
        }
    }
    if (audio.samples.empty()) throw std::runtime_error("Empty or invalid audio");
    return audio;
}
} // namespace

Audio decode_audio(const std::string &path) {
    drwav wav{};
    if (drwav_init_file(&wav, path.c_str(), nullptr)) {
        const auto close = [](drwav *decoder) { drwav_uninit(decoder); };
        const std::unique_ptr<drwav, decltype(close)> guard(&wav, close);
        auto audio = read_mono(wav.channels, wav.sampleRate, [&](size_t count, float *out) {
            return drwav_read_pcm_frames_f32(&wav, count, out);
        });
        if (audio.samples.size() != wav.totalPCMFrameCount)
            throw std::runtime_error("Truncated WAV audio: " + path);
        return audio;
    }
    const std::unique_ptr<drflac, decltype(&drflac_close)> flac(
        drflac_open_file(path.c_str(), nullptr), drflac_close);
    if (flac) {
        auto audio = read_mono(flac->channels, flac->sampleRate, [&](size_t count, float *out) {
            return drflac_read_pcm_frames_f32(flac.get(), count, out);
        });
        // FLAC permits an unknown (zero) total sample count in STREAMINFO.
        if (flac->totalPCMFrameCount && audio.samples.size() != flac->totalPCMFrameCount)
            throw std::runtime_error("Truncated FLAC audio: " + path);
        return audio;
    }
    drmp3 mp3{};
    if (drmp3_init_file(&mp3, path.c_str(), nullptr)) {
        const auto close = [](drmp3 *decoder) { drmp3_uninit(decoder); };
        const std::unique_ptr<drmp3, decltype(close)> guard(&mp3, close);
        return read_mono(mp3.channels, mp3.sampleRate, [&](size_t count, float *out) {
            return drmp3_read_pcm_frames_f32(&mp3, count, out);
        });
    }
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
