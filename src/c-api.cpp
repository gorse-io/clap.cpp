#include "clap.h"
#include "model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace {
thread_local char diagnostic[1024] = {};
struct InvalidArgument {
    const char *message;
};
void require(bool valid, const char *message) {
    if (!valid)
        throw InvalidArgument{message};
}
template <typename F> clap_status protect(F &&f) noexcept {
    diagnostic[0] = '\0';
    try {
        f();
        return CLAP_OK;
    } catch (const InvalidArgument &e) {
        std::snprintf(diagnostic, sizeof(diagnostic), "%s", e.message);
        return CLAP_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        std::snprintf(diagnostic, sizeof(diagnostic), "%s", "CLAP: out of memory");
        return CLAP_OUT_OF_MEMORY;
    } catch (const std::exception &e) {
        std::snprintf(diagnostic, sizeof(diagnostic), "%s", e.what());
    } catch (...) {
        std::snprintf(diagnostic, sizeof(diagnostic), "%s", "CLAP: unknown error");
    }
    return CLAP_ERROR;
}
void *allocate(size_t count, size_t size) {
    if (count > std::numeric_limits<size_t>::max() / size)
        throw std::bad_alloc();
    void *p = std::calloc(count, size);
    if (!p && count)
        throw std::bad_alloc();
    return p;
}
template <typename T> T *copy(const std::vector<T> &v) {
    T *p = static_cast<T *>(allocate(v.size(), sizeof(T)));
    if (!v.empty())
        std::memcpy(p, v.data(), v.size() * sizeof(T));
    return p;
}
char *copy_string(const std::string &s) {
    char *p = static_cast<char *>(allocate(s.size() + 1, 1));
    std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}
template <typename T> void reset(T **p, size_t *n) {
    if (p)
        *p = nullptr;
    if (n)
        *n = 0;
}
template <typename F> clap_status embedding(CLAP *model, float **out, size_t *count, F &&f) {
    reset(out, count);
    return protect([&] {
        require(model && out && count, "CLAP: model and output slots are required");
        const auto values = f();
        *out = copy(values);
        *count = values.size();
    });
}
std::vector<float> waveform(const float *pcm, size_t count, int rate) {
    require(pcm && count && rate > 0, "CLAP: nonempty PCM and a positive sample rate are required");
    require(count <= std::vector<float>().max_size(), "CLAP: too many samples");
    for (size_t i = 0; i < count; ++i)
        require(std::isfinite(pcm[i]), "CLAP: PCM must be finite");
    return {pcm, pcm + count};
}
} // namespace

extern "C" {
clap_runtime_options clap_default_options(void) { return {0, nullptr, CLAP_ACTIVATION_MODEL_DEFAULT}; }
const char *clap_last_error(void) { return diagnostic; }
clap_status clap_create(const char *path, const clap_runtime_options *options, CLAP **out) {
    if (out)
        *out = nullptr;
    return protect([&] {
        require(path && *path && out, "CLAP: nonempty model path and output slot are required");
        clap::RuntimeOptions native;
        if (options) {
            require(options->threads >= 0, "CLAP: threads must be nonnegative");
            require(options->activation_precision == CLAP_ACTIVATION_MODEL_DEFAULT ||
                        options->activation_precision == CLAP_ACTIVATION_F32,
                    "CLAP: invalid activation precision");
            native.threads = options->threads;
            if (options->device)
                native.device = options->device;
            if (options->activation_precision == CLAP_ACTIVATION_F32)
                native.activation_precision = clap::ActivationPrecision::F32;
        }
        *out = new CLAP(path, native);
    });
}
clap_status clap_create_with_mmproj(const char *path, const char *mmproj_path, const clap_runtime_options *options,
                                    CLAP **out) {
    if (out)
        *out = nullptr;
    return protect([&] {
        require(path && *path && out, "CLAP: nonempty model path and output slot are required");
        require(mmproj_path && *mmproj_path, "CLAP: nonempty mmproj path is required");
        clap::RuntimeOptions native;
        if (options) {
            require(options->threads >= 0, "CLAP: threads must be nonnegative");
            require(options->activation_precision == CLAP_ACTIVATION_MODEL_DEFAULT ||
                        options->activation_precision == CLAP_ACTIVATION_F32,
                    "CLAP: invalid activation precision");
            native.threads = options->threads;
            if (options->device)
                native.device = options->device;
            if (options->activation_precision == CLAP_ACTIVATION_F32)
                native.activation_precision = clap::ActivationPrecision::F32;
        }
        *out = new CLAP(path, native, mmproj_path);
    });
}
void clap_destroy(CLAP *model) { delete model; }
void clap_free(void *allocation) { std::free(allocation); }
void clap_free_devices(clap_device *devices, size_t count) {
    if (!devices)
        return;
    for (size_t i = 0; i < count; ++i) {
        std::free(const_cast<char *>(devices[i].name));
        std::free(const_cast<char *>(devices[i].description));
    }
    std::free(devices);
}
clap_status clap_available_devices(clap_device **out, size_t *count) {
    reset(out, count);
    return protect([&] {
        require(out && count, "CLAP: device output slots are required");
        const auto native = clap::available_devices();
        auto *devices = static_cast<clap_device *>(allocate(native.size(), sizeof(clap_device)));
        try {
            for (size_t i = 0; i < native.size(); ++i) {
                devices[i].name = copy_string(native[i].name);
                devices[i].description = copy_string(native[i].description);
                devices[i].gpu = native[i].gpu;
            }
        } catch (...) {
            clap_free_devices(devices, native.size());
            throw;
        }
        *out = devices;
        *count = native.size();
    });
}
void clap_free_audio_frames(clap_audio_frames *frames) {
    if (frames) {
        std::free(frames->values);
        *frames = {};
    }
}
clap_status clap_tokenize(const CLAP *model, const char *utf8, size_t max_length, int **out, size_t *count) {
    reset(out, count);
    return protect([&] {
        require(model && utf8 && out && count && max_length > 0,
                "CLAP: model, text, positive max_length and output slots are required");
        const auto tokens = clap::detail::tokenize(*model, utf8, max_length);
        *out = copy(tokens);
        *count = tokens.size();
    });
}
clap_status clap_encode_text(CLAP *model, const char *utf8, float **out, size_t *count) {
    return embedding(model, out, count, [&] {
        require(utf8, "CLAP: text is required");
        return clap::detail::encode_text(*model, utf8);
    });
}
clap_status clap_encode_tokens(CLAP *model, const int *tokens, size_t n, float **out, size_t *count) {
    return embedding(model, out, count, [&] {
        require(tokens && n && n <= std::vector<int>().max_size(), "CLAP: nonempty tokens are required");
        return model->architecture->text(std::vector<int>(tokens, tokens + n));
    });
}
clap_status clap_encode_audio(CLAP *model, const float *pcm, size_t n, int rate, size_t crop, float **out,
                              size_t *count) {
    return embedding(model, out, count,
                     [&] { return clap::detail::encode_audio(*model, waveform(pcm, n, rate), rate, crop); });
}
clap_status clap_encode_audio_file(CLAP *model, const char *path, size_t crop, float **out, size_t *count) {
    return embedding(model, out, count, [&] {
        require(path && *path, "CLAP: audio path is required");
        return clap::detail::encode_audio_file(*model, path, crop);
    });
}
clap_status clap_get_audio_frames(CLAP *model, const float *pcm, size_t n, int rate, clap_audio_frames *out) {
    if (out)
        *out = {};
    return protect([&] {
        require(model && out, "CLAP: model and frame output slot are required");
        const auto frames = clap::detail::audio_frames(*model, waveform(pcm, n, rate), rate);
        out->values = copy(frames.values);
        out->frames = static_cast<size_t>(frames.frames);
        out->dimension = static_cast<size_t>(frames.dimension);
    });
}
int clap_audio_sample_rate(const CLAP *model) { return model ? model->architecture->sample_rate() : 0; }
int clap_projection_dimension(const CLAP *model) { return model ? model->architecture->projection_dimension() : 0; }
} // extern "C"
