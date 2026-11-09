#include "audio.h"
#include "bench.h"
#include "clap.h"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void check(clap_status status) {
    if (status != CLAP_OK)
        throw std::runtime_error(clap_last_error());
}

// The native file reporter owns an internal stream which would otherwise be
// destroyed without checking late write/flush errors (for example /dev/full).
class CheckedReporter final : public benchmark::BenchmarkReporter {
  public:
    CheckedReporter(std::unique_ptr<benchmark::BenchmarkReporter> delegate, bool &failed)
        : delegate_(std::move(delegate)), failed_(failed) {}

    bool ReportContext(const Context &context) override {
        delegate_->SetOutputStream(&GetOutputStream());
        delegate_->SetErrorStream(&GetErrorStream());
        return delegate_->ReportContext(context);
    }
    void ReportRunsConfig(double min_time, bool explicit_iters, benchmark::IterationCount iters) override {
        delegate_->ReportRunsConfig(min_time, explicit_iters, iters);
    }
    void ReportRuns(const std::vector<Run> &runs) override { delegate_->ReportRuns(runs); }
    void Finalize() override {
        delegate_->Finalize();
        GetOutputStream().flush();
        GetErrorStream().flush();
        failed_ = failed_ || !GetOutputStream() || !GetErrorStream();
    }

  private:
    std::unique_ptr<benchmark::BenchmarkReporter> delegate_;
    bool &failed_;
};

std::string native_option(const BenchOptions &options, const std::string &name, const char *environment,
                          const char *fallback, bool boolean = false) {
    const char *value = std::getenv(environment);
    std::string result = value ? value : fallback;
    const auto flag = "--" + name;
    const auto prefix = flag + "=";
    for (const auto &argument : options.benchmark_args) {
        if (argument.starts_with(prefix))
            result = argument.substr(prefix.size());
        else if (boolean && argument == flag)
            result = "true";
    }
    return result;
}

bool native_tabular(const BenchOptions &options) {
    auto value = native_option(options, "benchmark_counters_tabular", "BENCHMARK_COUNTERS_TABULAR", "false", true);
    // Match the pinned library's boolean syntax, including its one-character aliases.
    if (value.size() == 1)
        return std::isalnum(static_cast<unsigned char>(value[0])) &&
               std::string("0fFnN").find(value[0]) == std::string::npos;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value != "false" && value != "no" && value != "off";
}

struct BenchmarkSession {
    ~BenchmarkSession() {
        benchmark::ClearRegisteredBenchmarks();
        benchmark::Shutdown();
    }
};
} // namespace

int cmd_bench(const BenchOptions &options) {
    // Defaults are native flags, not registration overrides: later user flags
    // retain all of Google Benchmark's parsing and override semantics.
    std::vector<std::string> arguments{"clap bench", "--benchmark_min_time=1s", "--benchmark_repetitions=3"};
    arguments.insert(arguments.end(), options.benchmark_args.begin(), options.benchmark_args.end());
    std::vector<char *> argv;
    for (auto &argument : arguments)
        argv.push_back(argument.data());
    int argc = static_cast<int>(argv.size());
    argv.push_back(nullptr);
    benchmark::Initialize(&argc, argv.data());
    const BenchmarkSession session;
    if (benchmark::ReportUnrecognizedArguments(argc, argv.data()))
        return 1;

    bool failed = false;
    bool prepared = false;
    std::string failure;
    std::unique_ptr<CLAP, decltype(&clap_destroy)> model(nullptr, clap_destroy);
    clap::Audio decoded;
    auto infer = [&] {
        if (options.frames) {
            clap_audio_frames sequence{};
            const std::unique_ptr<clap_audio_frames, decltype(&clap_free_audio_frames)> owned(&sequence,
                                                                                              clap_free_audio_frames);
            check(clap_get_audio_frames(model.get(), decoded.samples.data(), decoded.samples.size(),
                                        decoded.sample_rate, &sequence));
            benchmark::DoNotOptimize(sequence.values);
            benchmark::DoNotOptimize(sequence.frames);
            benchmark::ClobberMemory();
        } else {
            float *values = nullptr;
            std::size_t count = 0;
            const auto status = options.audio_path.empty()
                                    ? clap_encode_text(model.get(), options.text.c_str(), &values, &count)
                                    : clap_encode_audio(model.get(), decoded.samples.data(), decoded.samples.size(),
                                                        decoded.sample_rate, options.crop_offset, &values, &count);
            const std::unique_ptr<float, decltype(&clap_free)> owned(values, clap_free);
            check(status);
            benchmark::DoNotOptimize(values);
            benchmark::DoNotOptimize(count);
            benchmark::ClobberMemory();
        }
    };
    const char *name = options.frames ? "audio_frames" : options.audio_path.empty() ? "text" : "audio";
    benchmark::RegisterBenchmark(
        name,
        [&](benchmark::State &state) {
            try {
                // Callback setup precedes State's timer. It is deliberately lazy:
                // list-tests and unmatched filters never load/decode/infer anything.
                if (failed) {
                    state.SkipWithError(failure.c_str());
                    return;
                }
                if (!prepared) {
                    if (!options.audio_path.empty())
                        decoded = clap::decode_audio(options.audio_path);
                    auto runtime = clap_default_options();
                    runtime.threads = options.threads;
                    runtime.device = options.device.c_str();
                    CLAP *handle = nullptr;
                    const auto status = clap_create(options.model_path.c_str(), &runtime, &handle);
                    model.reset(handle);
                    check(status);
                    if (options.crop_requested && clap_audio_sample_rate(model.get()) != 48000)
                        throw std::runtime_error("--crop-offset requires a CLAP model");
                    infer(); // Exactly one untimed warmup, shared across calibration/repetitions.
                    prepared = true;
                }
                for (auto _ : state) {
                    (void)_;
                    infer(); // Full public API, including preprocessing, allocation and free.
                }
                state.SetItemsProcessed(state.iterations());
            } catch (const std::exception &error) {
                failed = true;
                failure = error.what();
                state.SkipWithError(failure.c_str());
            }
        })
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    benchmark::AddCustomContext("model", options.model_path);
    benchmark::AddCustomContext("requested_threads", std::to_string(options.threads));
    const char *environment_device = std::getenv("CLAP_DEVICE");
    benchmark::AddCustomContext("requested_device",
                                options.device.empty()
                                    ? (environment_device && *environment_device ? environment_device : "auto")
                                    : options.device);
    benchmark::AddCustomContext("input_mode", name);

    bool output_failed = false;
    std::unique_ptr<CheckedReporter> file_reporter;
    if (!native_option(options, "benchmark_out", "BENCHMARK_OUT", "").empty()) {
        const auto format = native_option(options, "benchmark_out_format", "BENCHMARK_OUT_FORMAT", "json");
        std::unique_ptr<benchmark::BenchmarkReporter> delegate;
        if (format == "json")
            delegate = std::make_unique<benchmark::JSONReporter>();
        else if (format == "csv") {
            // Preserve the pinned library's native CSV flag despite deprecation.
            BENCHMARK_DISABLE_DEPRECATED_WARNING
            delegate = std::make_unique<benchmark::CSVReporter>();
            BENCHMARK_RESTORE_DEPRECATED_WARNING
        } else
            delegate = std::make_unique<benchmark::ConsoleReporter>(
                native_tabular(options) ? benchmark::ConsoleReporter::OO_Tabular : benchmark::ConsoleReporter::OO_None);
        file_reporter = std::make_unique<CheckedReporter>(std::move(delegate), output_failed);
    }
    const auto matches = benchmark::RunSpecifiedBenchmarks(nullptr, file_reporter.get());
    std::cout.flush();
    std::cerr.flush();
    if (output_failed)
        std::cerr << "Failed to write benchmark report\n";
    return matches && !failed && !output_failed && std::cout && std::cerr ? 0 : 1;
}
