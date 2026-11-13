#include "clap.h"

#include <stdint.h>

_Static_assert(_Generic(((clap_device *)0)->gpu, _Bool: 1, default: 0), "gpu must be a C boolean");
_Static_assert(_Generic(((clap_context_options *)0)->threads, int: 1, default: 0), "threads must be int");

int main(void) {
    clap_model *model = (clap_model *)(uintptr_t)1;
    CLAP *context = (CLAP *)(uintptr_t)1;
    if (clap_model_load(NULL, NULL, &model) != CLAP_INVALID_ARGUMENT || model != NULL)
        return 1;
    if (clap_context_create(NULL, NULL, &context) != CLAP_INVALID_ARGUMENT || context != NULL)
        return 2;
    if (clap_default_context_options().threads != 0)
        return 3;
    clap_model_free(NULL);
    clap_destroy(NULL);
    if (!clap_last_error()[0] || clap_projection_dimension(NULL) || clap_audio_sample_rate(NULL))
        return 4;
    return 0;
}
