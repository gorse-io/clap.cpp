// Verify the catch path can report OOM while every C++ allocation fails.
#include "clap.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <cstdio>
static std::atomic<bool> fail{false};
void *operator new(std::size_t size) {
    if (fail.load()) throw std::bad_alloc();
    if (void *p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
int main() {
    clap_model *model = nullptr;
    const auto params = clap_model_default_params();
    fail = true;
    const auto status = clap_model_load("/a/long/path/which/forces/string/allocation/missing.gguf", params, &model);
    const bool ok = status == CLAP_STATUS_OUT_OF_MEMORY && model == nullptr &&
                    std::strlen(clap_last_error()) > 0;
    fail = false;
    if (!ok) { std::fprintf(stderr, "OOM boundary failed: %d %s\n", status, clap_last_error()); return 1; }
    // Valid FFT input also allocates; failed calls must not partially update it.
    clap_complex values[] = {{1, 0}, {2, 0}};
    fail = true;
    const auto fft_status = clap_fft(values, 2);
    fail = false;
    if (fft_status != CLAP_STATUS_OUT_OF_MEMORY || values[0].real != 1 || values[1].real != 2) return 1;
    return 0;
}
