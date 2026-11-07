#include "graph.h"
#include <ggml-cpu.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>

// Observe the actual backend configuration without exposing a public API getter.
static int configured_threads = 0;
extern "C" void __real_ggml_backend_cpu_set_n_threads(ggml_backend_t, int);
extern "C" void __wrap_ggml_backend_cpu_set_n_threads(ggml_backend_t backend, int threads) {
    configured_threads = threads;
    __real_ggml_backend_cpu_set_n_threads(backend, threads);
}

int main() {
    try {
        const int automatic = int(std::max(1u, std::thread::hardware_concurrency()));
        clap::detail::ModelLoader loader(clap::RuntimeOptions{0, "cpu"});
        for (int requested : {0, -1, 1, 3, 129}) {
            clap::detail::GraphRuntime runtime(loader, requested);
            const int expected = requested > 0 ? requested : automatic;
            std::cout << "requested=" << requested << " configured=" << configured_threads << " expected=" << expected
                      << '\n';
            if (configured_threads != expected)
                throw std::runtime_error(
                    "backend CPU threads must use hardware concurrency unless explicitly overridden");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
