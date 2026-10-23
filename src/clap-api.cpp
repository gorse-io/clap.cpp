#include "clap.h"
#include "clap-internal.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>

struct clap_model {
    std::mutex mutex;
    clap::Model value;
    clap_model(const char *path, int threads) : value(path, threads) {}
};

namespace {
thread_local char error_message[1024] = {};

// In particular, do not construct std::string in a catch clause: reporting an
// allocation failure must succeed even if all further allocations also fail.
void set_error(const char *message) noexcept {
    size_t i = 0;
    if (message) {
        for (; i + 1 < sizeof(error_message) && message[i]; ++i)
            error_message[i] = message[i];
    }
    error_message[i] = '\0';
}
template<class F> clap_status boundary(F &&function) noexcept {
    error_message[0] = '\0';
    try {
        function();
        return CLAP_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("CLAP: out of memory");
        return CLAP_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument &e) {
        set_error(e.what());
        return CLAP_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception &e) {
        set_error(e.what());
        return CLAP_STATUS_ERROR;
    } catch (...) {
        set_error("CLAP: unknown internal error");
        return CLAP_STATUS_ERROR;
    }
}
void require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}
template<class T> void input(const T *data, size_t count) {
    require(data && count > 0 && count <= (size_t(1) << 30) / sizeof(T),
            "CLAP: input is null, empty or larger than 1 GiB");
}
std::vector<float> waveform(const float *pcm, size_t count, int32_t rate) {
    input(pcm, count);
    require(rate > 0, "CLAP: sample rate must be positive");
    for (size_t i = 0; i < count; ++i)
        require(std::isfinite(pcm[i]), "CLAP: non-finite PCM sample");
    return {pcm, pcm + count};
}
template<class T, class U> T *copy(const std::vector<U> &values) {
    if (values.empty()) return nullptr;
    if (values.size() > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_alloc();
    auto *data = static_cast<T *>(std::malloc(values.size() * sizeof(T)));
    if (!data) throw std::bad_alloc();
    // All instantiations copy arithmetic types and cannot throw.
    for (size_t i = 0; i < values.size(); ++i) data[i] = static_cast<T>(values[i]);
    return data;
}
void store(const std::vector<float> &values, clap_float_buffer *out) {
    *out = {copy<float>(values), values.size()};
}
void store(const clap::Audio &audio, clap_audio *out) {
    *out = {copy<float>(audio.samples), audio.samples.size(), audio.sample_rate};
}
void store(const clap::AudioFrames &frames, clap_audio_frames *out) {
    *out = {copy<float>(frames.values), frames.values.size(), frames.frames, frames.dimension};
}
} // namespace

const char *clap_last_error(void) { return error_message; }
clap_model_params clap_model_default_params(void) { return {0}; }
clap_audio_params clap_audio_default_params(void) { return {48000, 0}; }

clap_status clap_model_load(const char *path, clap_model_params params, clap_model **out) {
    if (out) *out = nullptr;
    return boundary([&] {
        require(out && path && path[0], "CLAP: model path and output are required");
        require(params.n_threads >= 0, "CLAP: thread count must not be negative");
        *out = new clap_model(path, params.n_threads);
    });
}
void clap_model_free(clap_model *model) { delete model; }
clap_status clap_model_audio_sample_rate(clap_model *model, int32_t *out) {
    if (out) *out = 0;
    return boundary([&] {
        require(model && out, "CLAP: model and output are required");
        std::lock_guard<std::mutex> lock(model->mutex);
        *out = model->value.audio_sample_rate();
    });
}
clap_status clap_model_projection_dimension(clap_model *model, int32_t *out) {
    if (out) *out = 0;
    return boundary([&] {
        require(model && out, "CLAP: model and output are required");
        std::lock_guard<std::mutex> lock(model->mutex);
        *out = model->value.projection_dimension();
    });
}
clap_status clap_model_tokenize(clap_model *model, const char *utf8, size_t max_length, clap_token_buffer *out) {
    if (out) *out = {};
    return boundary([&] {
        require(model && out && utf8, "CLAP: model, text and output are required");
        require(max_length > 0 && max_length <= (size_t(1) << 28), "CLAP: invalid maximum token length");
        std::lock_guard<std::mutex> lock(model->mutex);
        const auto ids = model->value.tokenize(utf8, max_length);
        *out = {copy<int32_t>(ids), ids.size()};
    });
}
clap_status clap_model_text_tokens(clap_model *model, const int32_t *tokens, size_t count, clap_float_buffer *out) {
    if (out) *out = {};
    return boundary([&] {
        require(model && out, "CLAP: model and output are required");
        input(tokens, count);
        std::lock_guard<std::mutex> lock(model->mutex);
        store(model->value.text(std::vector<int>(tokens, tokens + count)), out);
    });
}
clap_status clap_model_text(clap_model *model, const char *utf8, clap_float_buffer *out) {
    if (out) *out = {};
    return boundary([&] {
        require(model && out && utf8, "CLAP: model, text and output are required");
        std::lock_guard<std::mutex> lock(model->mutex);
        store(model->value.text(utf8), out);
    });
}
clap_status clap_model_audio(clap_model *model, const float *pcm, size_t count,
                            clap_audio_params params, clap_float_buffer *out) {
    if (out) *out = {};
    return boundary([&] {
        require(model && out, "CLAP: model and output are required");
        std::lock_guard<std::mutex> lock(model->mutex);
        store(model->value.audio(waveform(pcm, count, params.sample_rate), params.sample_rate, params.crop_offset), out);
    });
}
clap_status clap_model_audio_frames(clap_model *model, const float *pcm, size_t count,
                                   int32_t rate, clap_audio_frames *out) {
    if (out) *out = {};
    return boundary([&] {
        require(model && out, "CLAP: model and output are required");
        std::lock_guard<std::mutex> lock(model->mutex);
        store(model->value.audio_frames(waveform(pcm, count, rate), rate), out);
    });
}
clap_status clap_model_clamp3_features(clap_model *model, const float *pcm, size_t count,
                                      int32_t rate, clap_audio_frames *out) {
    if (out) *out = {};
    return boundary([&] {
        require(model && out, "CLAP: model and output are required");
        std::lock_guard<std::mutex> lock(model->mutex);
        store(model->value.clamp3_features(waveform(pcm, count, rate), rate), out);
    });
}
clap_status clap_audio_decode(const char *path, clap_audio *out) {
    if (out) *out = {};
    return boundary([&] {
        require(path && path[0] && out, "CLAP: audio path and output are required");
        store(clap::decode_audio(path), out);
    });
}
clap_status clap_audio_resample(const float *pcm, size_t count, int32_t source_rate,
                               int32_t target_rate, clap_audio *out) {
    if (out) *out = {};
    return boundary([&] {
        require(out && target_rate > 0, "CLAP: output and positive target rate are required");
        store(clap::resample_audio({waveform(pcm, count, source_rate), source_rate}, target_rate), out);
    });
}
clap_status clap_log_mel(const float *pcm, size_t count, clap_audio_params params, clap_float_buffer *out) {
    if (out) *out = {};
    return boundary([&] {
        require(out && params.sample_rate == 48000, "CLAP: output and 48 kHz PCM are required");
        store(clap::log_mel(waveform(pcm, count, params.sample_rate), params.sample_rate, params.crop_offset), out);
    });
}
clap_status clap_fft(clap_complex *values, size_t count) {
    return boundary([&] {
        input(values, count);
        require((count & (count - 1)) == 0, "CLAP: FFT length must be a power of two");
        std::vector<std::complex<double>> data;
        data.reserve(count);
        for (size_t i = 0; i < count; ++i) data.emplace_back(values[i].real, values[i].imag);
        clap::fft(data);
        for (size_t i = 0; i < count; ++i) values[i] = {data[i].real(), data[i].imag()};
    });
}
void clap_float_buffer_free(clap_float_buffer *buffer) {
    if (buffer) { std::free(buffer->data); *buffer = {}; }
}
void clap_token_buffer_free(clap_token_buffer *buffer) {
    if (buffer) { std::free(buffer->data); *buffer = {}; }
}
void clap_audio_free(clap_audio *audio) {
    if (audio) { std::free(audio->samples); *audio = {}; }
}
void clap_audio_frames_free(clap_audio_frames *frames) {
    if (frames) { std::free(frames->values); *frames = {}; }
}
