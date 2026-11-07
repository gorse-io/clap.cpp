#pragma once

#if defined(__linux__)
#include <filesystem>
#endif

namespace clap::detail {
// A failed physical-core query returns zero; selection then uses logical CPUs,
// or one thread if neither source is available. Positive overrides are exact.
int select_cpu_threads(int requested, int physical, unsigned logical);
int default_cpu_threads();
#if defined(__linux__)
int linux_physical_cores(const std::filesystem::path &cpu_root);
#endif
} // namespace clap::detail
