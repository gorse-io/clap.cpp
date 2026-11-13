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

TEST(ABI, CreateOutOfMemoryPreservesDiagnosticsForQueriesAndFrees) {
    auto *model = reinterpret_cast<CLAP *>(1);
    clap_status status;
    bool valid_error;
    bool diagnostic_preserved;
    bool null_properties;
    {
        AllocationFailureScope failure_scope;
        status = clap_create("unused.gguf", nullptr, &model);
        const char *error = clap_last_error();
        valid_error = std::strstr(error, "out of memory") != nullptr;
        char saved[1024];
        std::snprintf(saved, sizeof(saved), "%s", error);
        null_properties = clap_projection_dimension(nullptr) == 0 && clap_audio_sample_rate(nullptr) == 0;
        (void)clap_default_options();
        clap_destroy(nullptr);
        clap_free(nullptr);
        clap_free_devices(nullptr, 0);
        clap_free_audio_frames(nullptr);
        diagnostic_preserved = std::strcmp(saved, clap_last_error()) == 0;
    }
    EXPECT_EQ(CLAP_OUT_OF_MEMORY, status);
    EXPECT_EQ(nullptr, model);
    EXPECT_TRUE(valid_error);
    EXPECT_TRUE(null_properties);
    EXPECT_TRUE(diagnostic_preserved);
}

TEST(ABI, CreateOutputResetAndValidationWithoutAllocations) {
    clap_status missing_path_status;
    clap_status missing_output_status;
    clap_status invalid_options_status;
    bool valid_path_error;
    bool valid_output_error;
    bool valid_options_error;
    auto *model = reinterpret_cast<CLAP *>(1);
    auto *invalid = reinterpret_cast<CLAP *>(1);
    auto options = clap_default_options();
    options.threads = -1;
    {
        AllocationFailureScope failure_scope;
        missing_path_status = clap_create(nullptr, nullptr, &model);
        valid_path_error = *clap_last_error() != '\0';
        missing_output_status = clap_create("unused.gguf", nullptr, nullptr);
        valid_output_error = *clap_last_error() != '\0';
        invalid_options_status = clap_create("unused.gguf", &options, &invalid);
        valid_options_error = std::strstr(clap_last_error(), "threads must be nonnegative") != nullptr;
    }
    EXPECT_EQ(CLAP_INVALID_ARGUMENT, missing_path_status);
    EXPECT_EQ(nullptr, model);
    EXPECT_TRUE(valid_path_error);
    EXPECT_EQ(CLAP_INVALID_ARGUMENT, missing_output_status);
    EXPECT_TRUE(valid_output_error);
    EXPECT_EQ(CLAP_INVALID_ARGUMENT, invalid_options_status);
    EXPECT_EQ(nullptr, invalid);
    EXPECT_TRUE(valid_options_error);
}

TEST(ABI, CreateDiagnosticsAreThreadLocal) {
    CLAP *model = nullptr;
    ASSERT_EQ(CLAP_INVALID_ARGUMENT, clap_create(nullptr, nullptr, &model));
    char saved[1024];
    std::snprintf(saved, sizeof(saved), "%s", clap_last_error());
    clap_status worker_status = CLAP_OK;
    bool worker_initially_clear = false;
    bool worker_error_present = false;
    std::thread worker([&] {
        worker_initially_clear = *clap_last_error() == '\0';
        CLAP *worker_model = nullptr;
        auto options = clap_default_options();
        options.threads = -1;
        worker_status = clap_create("unused.gguf", &options, &worker_model);
        worker_error_present = std::strstr(clap_last_error(), "threads must be nonnegative") != nullptr;
    });
    worker.join();
    EXPECT_TRUE(worker_initially_clear);
    EXPECT_EQ(CLAP_INVALID_ARGUMENT, worker_status);
    EXPECT_TRUE(worker_error_present);
    EXPECT_STREQ(saved, clap_last_error());
}
