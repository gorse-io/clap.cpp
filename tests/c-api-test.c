/* This file must remain strict ISO C99, without C++/ggml headers. */
#include "clap.h"
#include "clap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, clap_last_error()); return 1; } } while (0)
int main(int argc, char **argv) {
    struct clap_model *model = NULL;
    struct clap_model_params mp = clap_model_default_params();
    struct clap_audio_params ap = clap_audio_default_params();
    struct clap_float_buffer result = {0};
    struct clap_token_buffer tokens = {0};
    struct clap_audio decoded = {0};
    struct clap_audio_frames frames = {0};
    struct clap_complex values[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    const float pcm[] = {0.25f, -0.5f, 0.75f};
    CHECK(mp.n_threads == 0 && ap.sample_rate == 48000 && ap.crop_offset == 0);
    CHECK(clap_model_load(NULL, mp, &model) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(model == NULL && strlen(clap_last_error()) > 0);
    CHECK(clap_model_load("/missing/clap-c-api.gguf", mp, &model) != CLAP_STATUS_SUCCESS);
    CHECK(model == NULL);
    CHECK(clap_model_load("unused", mp, NULL) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_model_text(NULL, "text", &result) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(result.data == NULL && result.size == 0);
    CHECK(clap_model_tokenize(NULL, "text", 512, &tokens) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_model_text_tokens(NULL, NULL, 0, &result) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_model_audio(NULL, pcm, 3, ap, &result) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_model_audio_frames(NULL, pcm, 3, 48000, &frames) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_model_clamp3_features(NULL, pcm, 3, 24000, &frames) == CLAP_STATUS_INVALID_ARGUMENT);
    { int32_t n = 99;
      CHECK(clap_model_projection_dimension(NULL, &n) == CLAP_STATUS_INVALID_ARGUMENT && n == 0);
      CHECK(clap_model_audio_sample_rate(NULL, &n) == CLAP_STATUS_INVALID_ARGUMENT && n == 0); }
    CHECK(clap_audio_decode(NULL, &decoded) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_audio_decode("/missing/audio.wav", &decoded) != CLAP_STATUS_SUCCESS);
    CHECK(decoded.samples == NULL && decoded.n_samples == 0 && decoded.sample_rate == 0);
    CHECK(clap_audio_resample(pcm, 3, 48000, 48000, &decoded) == CLAP_STATUS_SUCCESS);
    CHECK(decoded.n_samples == 3 && decoded.sample_rate == 48000);
    CHECK(memcmp(decoded.samples, pcm, sizeof(pcm)) == 0);
    CHECK(strcmp(clap_last_error(), "") == 0);
    clap_audio_free(&decoded); clap_audio_free(&decoded);
    CHECK(!decoded.samples && !decoded.n_samples && !decoded.sample_rate);
    CHECK(clap_audio_resample(NULL, 3, 48000, 48000, &decoded) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_audio_resample(pcm, SIZE_MAX, 48000, 48000, &decoded) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_audio_resample(pcm, 3, 48000, 0, &decoded) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_audio_resample(pcm, 3, 48000, 48000, NULL) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_audio_resample(pcm, 3, 0, 48000, &decoded) == CLAP_STATUS_INVALID_ARGUMENT);
    { float bad[] = {NAN};
      CHECK(clap_audio_resample(bad, 1, 48000, 48000, &decoded) == CLAP_STATUS_INVALID_ARGUMENT); }
    CHECK(clap_log_mel(NULL, 0, ap, &result) == CLAP_STATUS_INVALID_ARGUMENT);
    { const float silence[] = {0}; size_t i;
      CHECK(clap_log_mel(silence, 1, ap, &result) == CLAP_STATUS_SUCCESS);
      CHECK(result.size == 1001 * 64);
      for (i = 0; i < result.size; ++i) CHECK(fabsf(result.data[i] + 100) < 1e-5f); }
    clap_float_buffer_free(&result); clap_float_buffer_free(&result);
    CHECK(!result.data && !result.size);
    CHECK(clap_fft(values, 4) == CLAP_STATUS_SUCCESS);
    CHECK(values[0].real == 10 && values[0].imag == 0);
    CHECK(fabs(values[1].real + 2) < 1e-12 && fabs(values[1].imag - 2) < 1e-12);
    CHECK(clap_fft(values, 3) == CLAP_STATUS_INVALID_ARGUMENT);
    CHECK(clap_fft(NULL, 4) == CLAP_STATUS_INVALID_ARGUMENT);
    clap_model_free(NULL); clap_float_buffer_free(NULL); clap_token_buffer_free(NULL);
    clap_audio_free(NULL); clap_audio_frames_free(NULL);
    clap_token_buffer_free(&tokens); clap_token_buffer_free(&tokens);
    clap_audio_frames_free(&frames); clap_audio_frames_free(&frames);
    if (argc > 1) {
        int32_t dim = 0, rate = 0;
        CHECK(clap_model_load(argv[1], mp, &model) == CLAP_STATUS_SUCCESS);
        CHECK(clap_model_projection_dimension(model, &dim) == CLAP_STATUS_SUCCESS && dim > 0);
        CHECK(clap_model_audio_sample_rate(model, &rate) == CLAP_STATUS_SUCCESS && rate > 0);
        CHECK(clap_model_tokenize(model, "hello", 512, &tokens) == CLAP_STATUS_SUCCESS);
        CHECK(tokens.size > 0);
        CHECK(clap_model_text_tokens(model, tokens.data, tokens.size, &result) == CLAP_STATUS_SUCCESS);
        CHECK(result.size == (size_t)dim);
        clap_float_buffer_free(&result); clap_token_buffer_free(&tokens);
        CHECK(clap_model_text(model, "hello", &result) == CLAP_STATUS_SUCCESS);
        CHECK(result.size == (size_t)dim);
        clap_float_buffer_free(&result); clap_model_free(model);
    }
    puts("strict C99 API consumer passed");
    return 0;
}
