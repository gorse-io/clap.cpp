#include "cpu-threads.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#if defined(__linux__)
#include <fstream>
#endif

static void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
int main(int argc, char **argv) {
    try {
        using namespace clap::detail;
        require(select_cpu_threads(0, 24, 48) == 24, "physical count must not be capped at eight");
        require(select_cpu_threads(-1, 24, 48) == 24, "nonpositive threads select automatic");
        for (int threads : {1, 3, 129})
            require(select_cpu_threads(threads, 24, 48) == threads, "explicit threads must be preserved");
        require(select_cpu_threads(0, 0, 48) == 48, "missing topology must use logical count without a cap");
        require(select_cpu_threads(0, -1, 0) == 1, "unavailable CPU counts must use one thread");
        require(select_cpu_threads(0, 0, std::numeric_limits<unsigned>::max()) == std::numeric_limits<int>::max(),
                "logical fallback must fit a positive int");
        require(default_cpu_threads() > 0, "host detection must return a usable count");
#if defined(__linux__)
        require(argc == 2, "cpu-threads-test FIXTURE_DIR");
        const std::filesystem::path root(argv[1]);
        std::filesystem::remove_all(root);
        auto cpu = [&](int id, int package, int core) {
            const auto dir = root / ("cpu" + std::to_string(id)) / "topology";
            std::filesystem::create_directories(dir);
            std::ofstream(dir / "physical_package_id") << package;
            std::ofstream(dir / "core_id") << core;
        };
        require(linux_physical_cores(root) == 0, "missing sysfs must signal fallback");
        for (int i = 0; i < 48; ++i)
            cpu(i, i / 24, (i % 24) / 2);
        require(linux_physical_cores(root) == 24, "deduplicate SMT siblings, not equal core IDs across sockets");
        cpu(48, 2, 0);
        std::ofstream(root / "cpu48/online") << 0;
        require(linux_physical_cores(root) == 24, "offline CPUs must not be counted");
        std::ofstream(root / "cpu48/online") << "invalid";
        require(linux_physical_cores(root) == 0, "malformed online status must signal fallback");
        std::ofstream(root / "cpu48/online") << 0;
        std::filesystem::create_directories(root / "cpufreq");
        require(linux_physical_cores(root) == 24, "ignore non-CPU entries");
        std::ofstream(root / "cpu0/topology/core_id") << "invalid";
        require(linux_physical_cores(root) == 0, "malformed topology must not return a partial count");
        cpu(0, 0, -1);
        require(linux_physical_cores(root) == 0, "unknown topology must signal fallback");
        std::filesystem::remove(root / "cpu0/topology/core_id");
        require(linux_physical_cores(root) == 0, "missing core ID must signal fallback");
        std::filesystem::remove_all(root);
#else
        (void)argc;
        (void)argv;
#endif
        std::cout << "CPU physical-core detection, overrides and fallbacks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
