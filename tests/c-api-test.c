#include "clap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, clap_last_error());                            \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

_Static_assert(_Generic(((clap_device *)0)->gpu, _Bool: 1, default: 0), "device gpu must be bool");

int main(int argc, char **argv) {
    CLAP *model = NULL;
    clap_runtime_options options = clap_default_options();
    clap_device *devices = NULL;
    size_t count = 0;
    int rate = 0;
    int *tokens = NULL;
    float *text = NULL, *encoded = NULL, *audio = NULL, *path = NULL;
    size_t ntokens = 0, ntext = 0, nencoded = 0, naudio = 0, npath = 0;
    clap_audio_frames frames = {0};
    float pcm[4800] = {0};
    CHECK(clap_create(NULL, NULL, &model) == CLAP_INVALID_ARGUMENT);
    CHECK(model == NULL && strlen(clap_last_error()) > 0);
    CHECK(clap_create("/missing/clap.gguf", NULL, &model) == CLAP_ERROR);
    CHECK(model == NULL);
    CHECK(clap_available_devices(&devices, &count) == CLAP_OK);
    CHECK(count > 0 && devices[0].name && devices[0].description);
    clap_free_devices(devices, count);
    CHECK(argc == 4);
    options.threads = -1;
    CHECK(clap_create(argv[1], &options, &model) == CLAP_INVALID_ARGUMENT && model == NULL);
    options.threads = 0;
    options.activation_precision = (clap_activation_precision)99;
    CHECK(clap_create(argv[1], &options, &model) == CLAP_INVALID_ARGUMENT && model == NULL);
    CHECK(clap_available_devices(NULL, &count) == CLAP_INVALID_ARGUMENT && count == 0);
    char diagnostic[1024];
    snprintf(diagnostic, sizeof(diagnostic), "%s", clap_last_error());
    CHECK(diagnostic[0] != '\0');
    CHECK(clap_projection_dimension(NULL) == 0);
    CHECK(clap_audio_sample_rate(NULL) == 0);
    CHECK(strcmp(clap_last_error(), diagnostic) == 0);
    options.device = "cpu";
    options.threads = 1;
    options.activation_precision = CLAP_ACTIVATION_F32;
    CHECK(clap_create(argv[1], &options, &model) == CLAP_OK);
    CHECK(clap_projection_dimension(model) == 5);
    rate = clap_audio_sample_rate(model);
    CHECK(rate == 48000);
    CHECK(clap_last_error()[0] == '\0');
    CHECK(clap_available_devices(NULL, &count) == CLAP_INVALID_ARGUMENT);
    snprintf(diagnostic, sizeof(diagnostic), "%s", clap_last_error());
    CHECK(clap_projection_dimension(model) == 5);
    CHECK(clap_audio_sample_rate(model) == rate);
    CHECK(strcmp(clap_last_error(), diagnostic) == 0);
    CHECK(clap_tokenize(model, "hello", 512, &tokens, &ntokens) == CLAP_OK && ntokens > 2);
    CHECK(clap_encode_text(model, "hello", &text, &ntext) == CLAP_OK && ntext == 5);
    CHECK(clap_encode_tokens(model, tokens, ntokens, &encoded, &nencoded) == CLAP_OK && nencoded == ntext);
    for (size_t i = 0; i < ntext; ++i)
        CHECK(fabsf(text[i] - encoded[i]) < 1e-6f);
    CHECK(clap_encode_audio(model, pcm, 4800, rate, 0, &audio, &naudio) == CLAP_OK && naudio == 5);
    CHECK(clap_encode_audio_file(model, argv[2], 0, &path, &npath) == CLAP_OK && npath == naudio);
    for (size_t i = 0; i < naudio; ++i)
        CHECK(fabsf(audio[i] - path[i]) < 1e-6f);
    CHECK(clap_get_audio_frames(model, pcm, 4800, rate, &frames) == CLAP_ERROR);
    CHECK(frames.values == NULL && frames.frames == 0 && frames.dimension == 0);
    {
        float *bad = (float *)1;
        size_t nbad = 99;
        CHECK(clap_encode_audio(model, NULL, 1, rate, 0, &bad, &nbad) == CLAP_INVALID_ARGUMENT);
        CHECK(bad == NULL && nbad == 0);
        CHECK(clap_encode_audio(model, pcm, 4800, 0, 0, &bad, &nbad) == CLAP_INVALID_ARGUMENT);
        CHECK(clap_encode_text(NULL, "hello", &bad, &nbad) == CLAP_INVALID_ARGUMENT);
        CHECK(clap_encode_text(model, NULL, &bad, &nbad) == CLAP_INVALID_ARGUMENT);
        CHECK(clap_encode_tokens(model, NULL, 0, &bad, &nbad) == CLAP_INVALID_ARGUMENT);
        CHECK(clap_encode_audio_file(model, NULL, 0, &bad, &nbad) == CLAP_INVALID_ARGUMENT);
        CHECK(clap_encode_audio_file(model, "/missing/audio.wav", 0, &bad, &nbad) == CLAP_ERROR);
        pcm[0] = NAN;
        CHECK(clap_encode_audio(model, pcm, 4800, rate, 0, &bad, &nbad) == CLAP_INVALID_ARGUMENT);
        pcm[0] = 0;
        CHECK(bad == NULL && nbad == 0);
        CHECK(clap_encode_text(model, "hello", NULL, &nbad) == CLAP_INVALID_ARGUMENT && nbad == 0);
        CHECK(clap_encode_text(model, "hello", &bad, NULL) == CLAP_INVALID_ARGUMENT && bad == NULL);
    }
    {
        int *bad = (int *)1;
        size_t nbad = 99;
        CHECK(clap_tokenize(model, "hello", 0, &bad, &nbad) == CLAP_INVALID_ARGUMENT);
        CHECK(bad == NULL && nbad == 0);
    }
    clap_free(tokens);
    clap_free(audio);
    clap_free(path);
    clap_free_audio_frames(&frames);
    clap_free_audio_frames(&frames);
    clap_destroy(model);
    for (size_t i = 0; i < ntext; ++i)
        CHECK(isfinite(text[i]) && fabsf(text[i] - encoded[i]) < 1e-6f);
    clap_free(text);
    clap_free(encoded);
    CHECK(clap_create(argv[3], &options, &model) == CLAP_OK);
    CHECK(clap_last_error()[0] == '\0');
    CHECK(clap_get_audio_frames(model, pcm, 4800, 24000, &frames) == CLAP_OK);
    CHECK(frames.values && frames.frames > 1 && frames.dimension == 4);
    audio = NULL;
    CHECK(clap_encode_audio(model, pcm, 4800, 24000, 0, &audio, &naudio) == CLAP_OK && naudio == 4);
    for (size_t column = 0; column < naudio; ++column) {
        float mean = 0;
        for (size_t row = 0; row < frames.frames; ++row)
            mean += frames.values[row * naudio + column] / (float)frames.frames;
        CHECK(fabsf(mean - audio[column]) < 1e-5f);
    }
    clap_free(audio);
    clap_free_audio_frames(&frames);
    CHECK(clap_encode_text(model, "hello", &text, &ntext) == CLAP_ERROR && text == NULL && ntext == 0);
    clap_destroy(model);
    clap_destroy(NULL);
    clap_free(NULL);
    clap_free_devices(NULL, 0);
    return 0;
}
