#include "clap.h"

#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <new>

namespace {
bool fail_allocations = false;

struct AllocationFailureScope {
    AllocationFailureScope() { fail_allocations = true; }
    ~AllocationFailureScope() { fail_allocations = false; }
};
} // namespace
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

TEST(CApiAllocation, ReportsOutOfMemoryWithoutAllocating) {
    CLAP *model = nullptr;
    clap_status status;
    bool valid_error;
    {
        AllocationFailureScope failure_scope;
        status = clap_create("unused.gguf", nullptr, &model);
        // Keep allocations disabled while reading the error: reporting must itself
        // be allocation-free, not replace bad_alloc with an escaping exception.
        valid_error = std::strstr(clap_last_error(), "out of memory") != nullptr;
        clap_destroy(nullptr);
        clap_free(nullptr);
    }
    // GoogleTest may allocate while reporting assertions, so restore allocations first.
    EXPECT_EQ(CLAP_OUT_OF_MEMORY, status);
    EXPECT_EQ(nullptr, model);
    EXPECT_TRUE(valid_error);
}
