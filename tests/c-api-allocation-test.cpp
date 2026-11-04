#include "clap.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace {
bool fail_allocations = false;
}
void *operator new(std::size_t size) {
    if (fail_allocations)
        throw std::bad_alloc();
    if (void *p = std::malloc(size ? size : 1))
        return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

int main() {
    CLAP *model = nullptr;
    fail_allocations = true;
    const auto status = clap_create("unused.gguf", nullptr, &model);
    // Keep allocations disabled while reading the error: reporting must itself
    // be allocation-free, not replace bad_alloc with an escaping exception.
    const bool valid =
        status == CLAP_OUT_OF_MEMORY && model == nullptr && std::strstr(clap_last_error(), "out of memory") != nullptr;
    clap_destroy(nullptr);
    clap_free(nullptr);
    fail_allocations = false;
    return valid ? 0 : 1;
}
