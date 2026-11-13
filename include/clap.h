/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

typedef struct CLAP CLAP;             /* Execution context. */
typedef struct clap_model clap_model; /* Shared immutable model. */

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

typedef struct clap_context_options {
    int threads; /* 0: inherit the model's thread default; otherwise positive. */
} clap_context_options;

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
 * Options and their strings are also borrowed for the call only.
 * Calls on the same handle must be serialized, including destruction. Distinct
 * handles may be used concurrently. No C++ exception crosses this API.
 * Allocating calls clear their output pointers/counts before validating inputs;
 * on failure those outputs remain zero/NULL. Pass fresh output slots:
 * overwriting a previous successful output does NOT free its allocation.
 * Fixed-dimension embedding calls instead write caller-owned storage, as below.
 * Each successful allocated output belongs to the caller, independently of the
 * model and context lifetimes. Use ONLY its matching free below (never the
 * C runtime free).
 * NULL is accepted by every free function. */
CLAP_API clap_runtime_options clap_default_options(void);
CLAP_API clap_context_options clap_default_context_options(void);
/* Diagnostic for the calling thread, valid until its next fallible API call.
 * Success clears it. Reading it, default-options queries, and destruction/free
 * functions never change it. */
CLAP_API const char *clap_last_error(void);
/* Load shared weights and fix device/activation precision. Runtime options have
 * the same semantics as clap_create. A successful model may create multiple
 * contexts with independent execution state; those contexts may run concurrently.
 * Freeing the public model handle does not invalidate existing contexts. */
CLAP_API clap_status clap_model_load(const char *gguf_path, const clap_runtime_options *options, clap_model **out);
CLAP_API void clap_model_free(clap_model *model);
/* NULL options or threads == 0 inherit the model's load-time thread default;
 * positive threads override it for this context; negative threads are invalid.
 * The model and output slot are required. Destroy contexts with clap_destroy. */
CLAP_API clap_status clap_context_create(const clap_model *model, const clap_context_options *options, CLAP **out);
/* Compatibility convenience: load a private model and create one context. */
CLAP_API clap_status clap_create(const char *gguf_path, const clap_runtime_options *options, CLAP **out);
CLAP_API void clap_destroy(CLAP *model);
CLAP_API clap_status clap_available_devices(clap_device **out, size_t *count);
/* Frees the list and its strings; count must be the original returned count. */
CLAP_API void clap_free_devices(clap_device *devices, size_t count);
CLAP_API void clap_free_audio_frames(clap_audio_frames *frames); /* Also zeroes the struct. */

/* Fixed-dimension embeddings: values is caller-owned, capacity counts float
 * elements (not bytes) and must be at least clap_projection_dimension(model).
 * Success writes exactly that many elements, leaving trailing capacity unchanged.
 * NULL values or insufficient capacity return CLAP_INVALID_ARGUMENT without
 * modifying storage. Other failures leave buffer contents unspecified.
 * The caller retains ownership and is responsible for releasing its storage.
 * Applies to all three encoders. */
CLAP_API clap_status clap_encode_text(CLAP *model, const char *utf8, float *values, size_t capacity);
/* Mono PCM, positive sample rate; automatically resampled. Nonempty finite
 * samples required. crop_offset is measured at the model sample rate: CLAP uses
 * a crop in long audio; other models may reject nonzero offsets. */
CLAP_API clap_status clap_encode_audio(CLAP *model, const float *pcm, size_t sample_count, int sample_rate,
                                       size_t crop_offset, float *values, size_t capacity);
/* Decode WAV/MP3/FLAC by contents, then encode with the same crop semantics. */
CLAP_API clap_status clap_encode_audio_file(CLAP *model, const char *path, size_t crop_offset, float *values,
                                            size_t capacity);
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
 * without final normalization and exposes no frames or crop offsets. */

#ifdef __cplusplus
}
#endif
#endif /* CLAP_H */
