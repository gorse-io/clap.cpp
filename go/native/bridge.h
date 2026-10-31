/* Private synchronous ABI for the Go binding. All strings are borrowed UTF-8.
 * Results own their vector and (until taken) model; free each result exactly once.
 * Callers must serialize operations and destruction on the same model.
 * No entrypoint permits a C++ exception to escape. Diagnostics live in results.
 */
#ifndef CLAP_GO_BRIDGE_H
#define CLAP_GO_BRIDGE_H
#include <stddef.h>
#ifdef __cplusplus
#define CG_NOEXCEPT noexcept
extern "C" {
#else
#define CG_NOEXCEPT
#endif
typedef struct cg_model cg_model;
typedef struct cg_result cg_result;
cg_result *cg_open(const char *path, int threads) CG_NOEXCEPT;
cg_result *cg_text(cg_model *, const char *) CG_NOEXCEPT;
cg_result *cg_audio(cg_model *, const char *, size_t crop) CG_NOEXCEPT;
const char *cg_error(const cg_result *) CG_NOEXCEPT;
cg_model *cg_take_model(cg_result *) CG_NOEXCEPT;
const float *cg_data(const cg_result *) CG_NOEXCEPT;
size_t cg_size(const cg_result *) CG_NOEXCEPT;
void cg_result_free(cg_result *) CG_NOEXCEPT;
void cg_model_free(cg_model *) CG_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#endif
