#include "clap.h"
#include "impl.h"
#include <algorithm>
#include <cmath>
#include <complex>

namespace clap {
using detail::check;
namespace {
constexpr double pi = 3.14159265358979323846;
void fft(std::vector<std::complex<double>> &a) {
    const size_t n = a.size();
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
double hz_to_mel(double hz) {
    return hz < 1000 ? hz / (200.0 / 3) : 15 + std::log(hz / 1000) / (std::log(6.4) / 27);
}
double mel_to_hz(double mel) {
    return mel < 15 ? mel * (200.0 / 3) : 1000 * std::exp((mel - 15) * (std::log(6.4) / 27));
}
} // namespace

std::vector<float> log_mel(const std::vector<float> &pcm, int sample_rate, size_t crop_offset) {
    check(sample_rate == 48000, "audio must be mono 48000 Hz (resample before inference)");
    check(!pcm.empty(), "empty waveform");
    check(std::all_of(pcm.begin(), pcm.end(), [](float x) { return std::isfinite(x); }), "non-finite waveform");
    constexpr size_t length = 480000;
    std::vector<double> wav(length, 0);
    if (pcm.size() > length) {
        check(crop_offset <= pcm.size() - length, "crop offset out of range");
        std::copy_n(pcm.data() + crop_offset, length, wav.data());
    } else {
        check(crop_offset == 0, "crop offset requires audio longer than 10 seconds");
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
} // namespace clap
