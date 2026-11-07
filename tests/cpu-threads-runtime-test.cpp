#include "cpu-threads.h"
#include "graph.h"
#include <ggml-cpu.h>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

// Observe the actual backend configuration without exposing a public API getter.
static int configured_threads = 0;
extern "C" void __real_ggml_backend_cpu_set_n_threads(ggml_backend_t, int);
extern "C" void __wrap_ggml_backend_cpu_set_n_threads(ggml_backend_t backend, int threads) {
    configured_threads = threads;
    __real_ggml_backend_cpu_set_n_threads(backend, threads);
}

int main() {
    try {
        // Independent Linux oracle: package/core pairs in /proc, not sysfs.
        std::ifstream input("/proc/cpuinfo");
        std::set<std::pair<int, int>> cores;
        int package = -1, core = -1;
        std::string line;
        auto record = [&] {
            if (package >= 0 && core >= 0)
                cores.emplace(package, core);
            package = core = -1;
        };
        while (std::getline(input, line)) {
            if (line.empty()) {
                record();
                continue;
            }
            const auto colon = line.find(':');
            if (colon == std::string::npos)
                continue;
            std::istringstream value(line.substr(colon + 1));
            if (line.compare(0, 11, "physical id") == 0)
                value >> package;
            if (line.compare(0, 7, "core id") == 0)
                value >> core;
        }
        record();
        if (cores.empty() || clap::detail::linux_physical_cores("/sys/devices/system/cpu") == 0) {
            std::cout << "Physical topology or independent /proc oracle unavailable; skipped\n";
            return 77;
        }
        clap::detail::ModelLoader loader(clap::RuntimeOptions{0, "cpu"});
        for (int requested : {0, -1, 1, 3, 129}) {
            clap::detail::GraphRuntime runtime(loader, requested);
            const int expected = requested > 0 ? requested : int(cores.size());
            std::cout << "requested=" << requested << " configured=" << configured_threads << " expected=" << expected
                      << '\n';
            if (configured_threads != expected)
                throw std::runtime_error("backend CPU threads must use physical cores unless explicitly overridden");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
