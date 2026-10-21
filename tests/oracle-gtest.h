#pragma once

#include <gtest/gtest.h>
#include <iostream>

// Oracle executables reserve stdout for their existing machine-readable protocol.
// Keep GoogleTest diagnostics on stderr without redirecting the process's stdout.
namespace oracle_test {
class StderrListener final : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo &info) override {
        std::cerr << "[ RUN      ] " << info.test_suite_name() << '.' << info.name() << '\n';
    }
    void OnTestPartResult(const testing::TestPartResult &result) override {
        if (result.passed()) return;
        if (result.file_name()) std::cerr << result.file_name() << ':' << result.line_number() << ": ";
        std::cerr << result.summary() << '\n';
    }
    void OnTestEnd(const testing::TestInfo &info) override {
        std::cerr << (info.result()->Skipped() ? "[  SKIPPED ] " :
                     info.result()->Passed() ? "[       OK ] " : "[  FAILED  ] ")
                  << info.test_suite_name() << '.' << info.name() << '\n';
    }
    void OnTestProgramEnd(const testing::UnitTest &tests) override {
        std::cerr << "[==========] " << tests.test_to_run_count() << " tests; "
                  << tests.successful_test_count() << " passed, "
                  << tests.skipped_test_count() << " skipped, "
                  << tests.failed_test_count() << " failed\n";
    }
};

inline void initialize(int *argc, char **argv) {
    testing::InitGoogleTest(argc, argv);
    auto &listeners = testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new StderrListener);
}
} // namespace oracle_test
