#include "cpu-threads.h"
#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <thread>

#if defined(__linux__)
#include <fstream>
#include <set>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vector>
#endif

namespace clap::detail {
int select_cpu_threads(int requested, int physical, unsigned logical) {
    if (requested > 0)
        return requested;
    if (physical > 0)
        return physical;
    return int(std::max(1u, std::min(logical, unsigned(std::numeric_limits<int>::max()))));
}

#if defined(__linux__)
int linux_physical_cores(const std::filesystem::path &cpu_root) {
    std::set<std::pair<int, int>> cores;
    std::error_code error;
    std::filesystem::directory_iterator it(cpu_root, error), end;
    for (; !error && it != end; it.increment(error)) {
        const auto name = it->path().filename().string();
        if (name.size() <= 3 || name.compare(0, 3, "cpu") != 0 ||
            name.find_first_not_of("0123456789", 3) != std::string::npos)
            continue;
        int online = 1;
        std::ifstream online_file(it->path() / "online");
        if (online_file.is_open() && (!(online_file >> online) || (online != 0 && online != 1)))
            return 0;
        if (!online)
            continue;
        int package = -1, core = -1;
        const bool have_package = bool(std::ifstream(it->path() / "topology/physical_package_id") >> package);
        const bool have_core = bool(std::ifstream(it->path() / "topology/core_id") >> core);
        // Never return a partial count if the kernel does not expose topology.
        if (!have_package || !have_core || package < 0 || core < 0)
            return 0;
        cores.emplace(package, core);
    }
    return error ? 0 : int(cores.size());
}
#endif

static int physical_cores() {
#if defined(__linux__)
    return linux_physical_cores("/sys/devices/system/cpu");
#elif defined(__APPLE__)
    int cores = 0;
    size_t size = sizeof(cores);
    return sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0) == 0 && size == sizeof(cores) ? cores : 0;
#elif defined(_WIN32)
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || !bytes)
        return 0;
    // The API returns variable-sized records, one per core across all groups.
    std::vector<unsigned char> buffer(bytes);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &bytes))
        return 0;
    int cores = 0;
    for (size_t offset = 0; offset < bytes;) {
        if (bytes - offset < offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor))
            return 0;
        const auto *info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer.data() + offset);
        if (info->Size < offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor) || info->Size > bytes - offset)
            return 0;
        if (info->Relationship == RelationProcessorCore)
            ++cores;
        offset += info->Size;
    }
    return cores;
#else
    return 0;
#endif
}

int default_cpu_threads() {
    int physical = 0;
    try {
        physical = physical_cores();
    } catch (const std::exception &) {
        // Missing topology, filesystem failures or allocation failure: retain a
        // usable automatic setting even when physical detection is unavailable.
    }
    return select_cpu_threads(0, physical, std::thread::hardware_concurrency());
}
} // namespace clap::detail
