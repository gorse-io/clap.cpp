#ifndef CLAP_H
#define CLAP_H

#include <stdbool.h>
#include <stddef.h>

/* Windows DLL clients define CLAP_SHARED; static clients need no defines. */
#if defined(_WIN32) && defined(CLAP_SHARED)
#if defined(CLAP_BUILD)
#define CLAP_API __declspec(dllexport)
#else
#define CLAP_API __declspec(dllimport)
#endif
#else
#define CLAP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CLAP CLAP;

typedef enum clap_status { CLAP_OK = 0, CLAP_INVALID_ARGUMENT = 1, CLAP_OUT_OF_MEMORY = 2, CLAP_ERROR = 3 } clap_status;

typedef enum clap_activation_precision {
    CLAP_ACTIVATION_MODEL_DEFAULT = 0,
    CLAP_ACTIVATION_F32 = 1
} clap_activation_precision;

typedef struct clap_runtime_options {
    int threads;        /* 0: automatic; otherwise positive. */
    const char *device; /* NULL/empty: CLAP_DEVICE or auto; cpu forces CPU. */
    clap_activation_precision activation_precision;
} clap_runtime_options;

typedef struct clap_device {
    const char *name;
    const char *description;
    bool gpu;
} clap_device;

typedef struct clap_audio_frames {
    float *values; /* Time-major, frames * dimension elements. */
    size_t frames;
    size_t dimension;
} clap_audio_frames;

/* All input strings are NUL-terminated UTF-8, borrowed for the call only.
 * Calls on the same handle must be serialized, including destruction. Distinct
 * handles may be used concurrently. No C++ exception crosses this API.
 * Fallible calls clear their output pointers/counts before validating inputs;
 * on failure all provided outputs remain zero/NULL. Pass fresh output slots:
 * overwriting a previous successful output does NOT free its allocation.
 * Each successful allocated output belongs to the caller, independently of the
 * model lifetime. Use ONLY its matching free below (never the C runtime free).
 * NULL is accepted by every free function. */
CLAP_API clap_runtime_options clap_default_options(void);
/* Diagnostic for the calling thread, valid until its next fallible API call.
 * Success clears it. Reading it and freeing outputs never change it. */
CLAP_API const char *clap_last_error(void);
CLAP_API clap_status clap_create(const char *gguf_path, const clap_runtime_options *options, CLAP **out);
CLAP_API void clap_destroy(CLAP *model);
CLAP_API clap_status clap_available_devices(clap_device **out, size_t *count);
/* Frees the list and its strings; count must be the original returned count. */
CLAP_API void clap_free_devices(clap_device *devices, size_t count);
CLAP_API void clap_free(void *allocation);                       /* Tokens and embedding buffers. */
CLAP_API void clap_free_audio_frames(clap_audio_frames *frames); /* Also zeroes the struct. */

CLAP_API clap_status clap_tokenize(const CLAP *model, const char *utf8, size_t max_length, int **tokens, size_t *count);
CLAP_API clap_status clap_encode_text(CLAP *model, const char *utf8, float **values, size_t *count);
CLAP_API clap_status clap_encode_tokens(CLAP *model, const int *tokens, size_t token_count, float **values,
                                        size_t *count);
/* Mono PCM, positive sample rate; automatically resampled. Nonempty finite
 * samples required. crop_offset is measured at the model sample rate: CLAP uses
 * a crop in long audio; other models may reject nonzero offsets. */
CLAP_API clap_status clap_encode_audio(CLAP *model, const float *pcm, size_t sample_count, int sample_rate,
                                       size_t crop_offset, float **values, size_t *count);
/* Decode WAV/MP3/FLAC by contents, then encode with the same crop semantics. */
CLAP_API clap_status clap_encode_audio_file(CLAP *model, const char *path, size_t crop_offset, float **values,
                                            size_t *count);
CLAP_API clap_status clap_get_audio_frames(CLAP *model, const float *pcm, size_t sample_count, int sample_rate,
                                           clap_audio_frames *out);
/* Read-only properties; return 0 for a NULL model or an unavailable property.
 * These queries do not change clap_last_error(). */
CLAP_API int clap_audio_sample_rate(const CLAP *model);
CLAP_API int clap_projection_dimension(const CLAP *model);

/* Architecture is selected from GGUF metadata. Unsupported modalities return
 * CLAP_ERROR. CLAP projections are unit-normalized; CLaMP3 uses native unnormalized
 * projections and requires a bundled MERT frontend. MERT/MERT2/MuQ audio embeddings
 * mean-pool final-layer frames; their frame API returns the unpooled sequence.
 * MuQ-MuLan text is L2-normalized; audio averages normalized 10-second clips
 * without final normalization and exposes no frames or crop offsets. Qwen3 text
 * pools the last nonpadding token and L2-normalizes it; audio is unsupported.
 * Qwen3 token input must be unpadded when the checkpoint shares EOS/padding IDs. */

#ifdef __cplusplus
}
#endif
#endif /* CLAP_H */
