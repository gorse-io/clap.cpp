#ifndef CLAP_H
#define CLAP_H

#include <stddef.h>
#include <stdint.h>

/* Define CLAP_SHARED when using a DLL outside the supplied CMake target.
 * CLAP_BUILD is reserved for building the library itself. Static users need
 * neither definition. No C++ or ggml headers are required by this interface. */
#ifndef CLAP_API
#  if defined(CLAP_SHARED) && defined(_WIN32)
#    if defined(CLAP_BUILD)
#      define CLAP_API __declspec(dllexport)
#    else
#      define CLAP_API __declspec(dllimport)
#    endif
#  elif defined(CLAP_SHARED) && defined(__GNUC__)
#    define CLAP_API __attribute__((visibility("default")))
#  else
#    define CLAP_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct clap_model;

enum clap_status {
    CLAP_STATUS_SUCCESS = 0,
    CLAP_STATUS_INVALID_ARGUMENT = 1,
    CLAP_STATUS_OUT_OF_MEMORY = 2,
    CLAP_STATUS_ERROR = 3
};

struct clap_model_params {
    int32_t n_threads; /* 0: automatic; negative values are invalid. */
};
struct clap_audio_params {
    int32_t sample_rate;
    size_t crop_offset; /* Input offset in samples at the model's native rate. */
};
struct clap_float_buffer { float *data; size_t size; };
struct clap_token_buffer { int32_t *data; size_t size; };
struct clap_audio { float *samples; size_t n_samples; int32_t sample_rate; };
/* Time-major values: n_values == frames * dimension. */
struct clap_audio_frames {
    float *values;
    size_t n_values;
    int32_t frames;
    int32_t dimension;
};
struct clap_complex { double real; double imag; };

/* Lifetime and concurrency contract:
 * - Strings are NUL-terminated UTF-8 (paths use the platform's native narrow
 *   path encoding). Inputs are borrowed for the duration of a call only.
 * - Initialize output structs to {0}. Status-returning functions reset outputs
 *   before validation, leave them empty on failure, and allocate on success.
 *   Free previous results BEFORE reusing an output. Inputs and outputs must
 *   not alias. Never free or resize library buffers with the caller's allocator.
 * - Use the matching *_free below. Free accepts NULL, resets buffer structs,
 *   and is repeatable on the same reset struct. Do not copy owning structs
 *   and free both copies. Model handles must be freed exactly once.
 * - All calls on the same model are serialized internally (including getters
 *   and tokenization). Different handles and standalone helpers may run in
 *   parallel. The caller must prevent model_free from overlapping any use of
 *   that handle, and must synchronize access to its own inputs/outputs.
 * - No C++ exception crosses this API. Errors are per calling thread, in a
 *   fixed-size buffer (messages may be truncated). clap_last_error() is borrowed
 *   until the next status-returning call on that thread or thread exit; success
 *   clears it.
 *   Default-parameter functions, frees and clap_last_error do not change it.
 * - NULL required pointers, nonpositive rates, non-finite PCM, empty or >1 GiB
 *   PCM/token/FFT inputs are invalid arguments. Unsupported model capabilities,
 *   malformed files and inference failures return CLAP_STATUS_ERROR. A valid
 *   pointer must address the entire declared input; dangling handles and invalid
 *   pointers cannot be diagnosed. */
CLAP_API const char *clap_last_error(void);
CLAP_API struct clap_model_params clap_model_default_params(void);
CLAP_API struct clap_audio_params clap_audio_default_params(void); /* 48000 Hz, crop 0 */

/* Native CLAP, CLaMP3, MERT-v1/v2, MuQ or MuQ-MuLan, selected by GGUF metadata.
 * Weights are memory-mapped where supported; retain the file for handle lifetime.
 * CLaMP3 requires its bundled MERT-v1 frontend. */
CLAP_API enum clap_status clap_model_load(const char *gguf_path,
    struct clap_model_params params, struct clap_model **out_model);
CLAP_API void clap_model_free(struct clap_model *model);
CLAP_API enum clap_status clap_model_audio_sample_rate(struct clap_model *model, int32_t *out_rate);
CLAP_API enum clap_status clap_model_projection_dimension(struct clap_model *model, int32_t *out_dimension);
/* Tokenization truncates according to the model tokenizer's max_length policy;
 * use 512 for the old default. Models without text encoders reject these calls.
 * The string inference call uses the model context (MuLan rejects overflow). */
CLAP_API enum clap_status clap_model_tokenize(struct clap_model *model, const char *utf8,
    size_t max_length, struct clap_token_buffer *out_tokens);
CLAP_API enum clap_status clap_model_text_tokens(struct clap_model *model, const int32_t *tokens,
    size_t n_tokens, struct clap_float_buffer *out_embedding);
CLAP_API enum clap_status clap_model_text(struct clap_model *model, const char *utf8,
    struct clap_float_buffer *out_embedding);
/* Mono PCM, automatically resampled. CLAP projections are unit-normalized;
 * CLaMP3 projections are unnormalized. MERT/MuQ mean-pool final frames.
 * MuLan averages individually normalized 10 s clips without final normalization.
 * Crop offsets are supported by CLAP only; waveform encoders process all input. */
CLAP_API enum clap_status clap_model_audio(struct clap_model *model, const float *pcm,
    size_t n_samples, struct clap_audio_params params, struct clap_float_buffer *out_embedding);
/* Final-layer waveform-encoder frames (MERT-v1/v2, MuQ); not joint CLAP/MuLan. */
CLAP_API enum clap_status clap_model_audio_frames(struct clap_model *model, const float *pcm,
    size_t n_samples, int32_t sample_rate, struct clap_audio_frames *out_frames);
/* Standalone MERT-v1 CLaMP3 preprocessing: whole-recording normalization,
 * non-overlapping 5 s windows, drop tails <1 s, mean of time and ALL layers.
 * At least one second is required; resamples to 24 kHz automatically. */
CLAP_API enum clap_status clap_model_clamp3_features(struct clap_model *model, const float *pcm,
    size_t n_samples, int32_t sample_rate, struct clap_audio_frames *out_frames);

/* Decode WAV/MP3/FLAC by contents, averaging channels, preserving source rate. */
CLAP_API enum clap_status clap_audio_decode(const char *path, struct clap_audio *out_audio);
/* Linear resampling with eighth-order anti-alias LPF, causal startup latency.
 * Output length ceil(n_samples * target_rate / source_rate); same-rate identity. */
CLAP_API enum clap_status clap_audio_resample(const float *pcm, size_t n_samples,
    int32_t source_rate, int32_t target_rate, struct clap_audio *out_audio);
/* 48 kHz PCM -> [1001,64] Slaney log-power mel. Short clips repeatpad;
 * long clips crop ten seconds at params.crop_offset. Rate must be 48000. */
CLAP_API enum clap_status clap_log_mel(const float *pcm, size_t n_samples,
    struct clap_audio_params params, struct clap_float_buffer *out_features);
/* In-place, unnormalized forward FFT with double intermediates. Nonzero
 * power-of-two length only. The input remains unchanged on failure. */
CLAP_API enum clap_status clap_fft(struct clap_complex *values, size_t count);

CLAP_API void clap_float_buffer_free(struct clap_float_buffer *buffer);
CLAP_API void clap_token_buffer_free(struct clap_token_buffer *buffer);
CLAP_API void clap_audio_free(struct clap_audio *audio);
CLAP_API void clap_audio_frames_free(struct clap_audio_frames *frames);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* CLAP_H */
