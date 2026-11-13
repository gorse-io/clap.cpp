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
#include "clap.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

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

TEST(ABI, OutOfMemory) {
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

TEST(ABI, ModelLoadOutOfMemory) {
    auto *model = reinterpret_cast<clap_model *>(1);
    clap_status status;
    bool valid_error;
    bool diagnostic_preserved;
    int default_threads;
    {
        AllocationFailureScope failure_scope;
        status = clap_model_load("unused.gguf", nullptr, &model);
        const char *error = clap_last_error();
        valid_error = std::strstr(error, "out of memory") != nullptr;
        char saved[1024];
        std::snprintf(saved, sizeof(saved), "%s", error);
        default_threads = clap_default_context_options().threads;
        (void)clap_default_options();
        clap_model_free(nullptr);
        clap_destroy(nullptr);
        clap_free(nullptr);
        clap_free_devices(nullptr, 0);
        clap_free_audio_frames(nullptr);
        diagnostic_preserved = std::strcmp(saved, clap_last_error()) == 0;
    }
    EXPECT_EQ(CLAP_OUT_OF_MEMORY, status);
    EXPECT_EQ(nullptr, model);
    EXPECT_TRUE(valid_error);
    EXPECT_EQ(0, default_threads);
    EXPECT_TRUE(diagnostic_preserved);
}

TEST(ABI, SharedModelOutputResetAndValidationWithoutAllocations) {
    clap_status model_status;
    clap_status context_status;
    bool valid_model_error;
    bool valid_context_error;
    auto *model = reinterpret_cast<clap_model *>(1);
    auto *context = reinterpret_cast<CLAP *>(1);
    {
        AllocationFailureScope failure_scope;
        model_status = clap_model_load(nullptr, nullptr, &model);
        valid_model_error = *clap_last_error() != '\0';
        context_status = clap_context_create(nullptr, nullptr, &context);
        valid_context_error = *clap_last_error() != '\0';
    }
    EXPECT_EQ(CLAP_INVALID_ARGUMENT, model_status);
    EXPECT_EQ(nullptr, model);
    EXPECT_TRUE(valid_model_error);
    EXPECT_EQ(CLAP_INVALID_ARGUMENT, context_status);
    EXPECT_EQ(nullptr, context);
    EXPECT_TRUE(valid_context_error);
}

TEST(ABI, SharedModelDiagnosticsAreThreadLocal) {
    clap_model *model = nullptr;
    ASSERT_EQ(CLAP_INVALID_ARGUMENT, clap_model_load(nullptr, nullptr, &model));
    char saved[1024];
    std::snprintf(saved, sizeof(saved), "%s", clap_last_error());
    clap_status worker_status = CLAP_OK;
    bool worker_initially_clear = false;
    bool worker_error_present = false;
    std::thread worker([&] {
        worker_initially_clear = *clap_last_error() == '\0';
        clap_model *worker_model = nullptr;
        auto options = clap_default_options();
        options.threads = -1;
        worker_status = clap_model_load("unused.gguf", &options, &worker_model);
        worker_error_present = std::strstr(clap_last_error(), "threads must be nonnegative") != nullptr;
    });
    worker.join();
    EXPECT_TRUE(worker_initially_clear);
    EXPECT_EQ(CLAP_INVALID_ARGUMENT, worker_status);
    EXPECT_TRUE(worker_error_present);
    EXPECT_STREQ(saved, clap_last_error());
}
