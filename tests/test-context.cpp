#include "audio.h"
#include "clap.h"
#include "context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CLAP_SHARED
namespace {
std::atomic<long> allocation_budget{-1};
std::atomic<size_t> allocation_count{0};
} // namespace
void *operator new(std::size_t bytes) {
    const auto remaining = allocation_budget.load();
    if (remaining == 0)
        throw std::bad_alloc();
    if (remaining > 0)
        allocation_budget.fetch_sub(1);
    allocation_count.fetch_add(1);
    if (auto *data = std::malloc(bytes ? bytes : 1))
        return data;
    throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *data) noexcept { std::free(data); }
void operator delete[](void *data) noexcept { std::free(data); }
void operator delete(void *data, std::size_t) noexcept { std::free(data); }
void operator delete[](void *data, std::size_t) noexcept { std::free(data); }
#endif

namespace {
void require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
void check(clap_status status) {
    if (status != CLAP_OK)
        throw std::runtime_error(clap_last_error());
}
using Model = std::unique_ptr<clap_model, decltype(&clap_model_free)>;
using Execution = std::unique_ptr<CLAP, decltype(&clap_destroy)>;
using Output = std::unique_ptr<float, decltype(&clap_free)>;
std::vector<float> encode(CLAP *context, const std::string &text) {
    float *raw = nullptr;
    size_t size = 0;
    check(clap_encode_text(context, text.c_str(), &raw, &size));
    Output output(raw, clap_free);
    return {raw, raw + size};
}
void compare(const std::vector<float> &actual, const std::vector<float> &expected) {
    require(actual.size() == expected.size(), "context output width differs from scalar oracle");
    for (size_t i = 0; i < actual.size(); ++i)
        require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 2e-5f,
                "context output differs from scalar oracle");
}
std::vector<float> rows(const ggml_tensor *table, const std::vector<int> &indices) {
    std::vector<float> result;
    const auto *data = static_cast<const float *>(table->data);
    for (const auto index : indices)
        result.insert(result.end(), data + index * table->ne[0], data + (index + 1) * table->ne[0]);
    return result;
}

// Both builders must enter while holding their respective runtime locks. A
// model-wide execution lock fails this bounded rendezvous, without a speed test.
void graph_isolation(CLAP *first, CLAP *second) {
    const auto &model = first->context.model();
    require(&model == &second->context.model(), "contexts did not retain the same immutable model");
    auto &a = first->context.runtime(model.architecture.get(), model.loader);
    auto &b = second->context.runtime(model.architecture.get(), model.loader);
    require(&a != &b, "execution runtimes are shared");
    require(a.stats().threads == 1 && b.stats().threads == 2, "context-local CPU thread override was lost");
    auto *table = model.loader.weight("text.embeddings.word_embeddings.weight");
    const auto *bytes = static_cast<const unsigned char *>(table->data);
    const std::vector<unsigned char> snapshot(bytes, bytes + ggml_nbytes(table));
    {
        clap::detail::Graph one(a), two(b);
        require(one.ctx() != two.ctx(), "graph metadata is shared");
        auto *x = one.weight(table, false), *y = two.weight(table, false);
        require(x == y && x->buffer == y->buffer, "model weight tensor/buffer was duplicated");
    }
    auto *positions = model.loader.weight("text.embeddings.position_embeddings.weight");
    const auto before = model.weights->stats();
    std::promise<void> entered_a, entered_b, release;
    auto ready_a = entered_a.get_future(), ready_b = entered_b.get_future();
    auto gate = release.get_future().share();
    const std::vector<int> left{2, 3, 4}, right{7, 8, 9};
    auto run = [&](clap::detail::GraphRuntime &runtime, const std::vector<int> &ids, std::promise<void> &entered) {
        return runtime.run_cached(
            "shared-context-barrier", {clap::detail::GraphInput(ids)}, [&](clap::detail::Graph &graph) {
                entered.set_value();
                gate.wait();
                return std::vector<ggml_tensor *>{graph.embedding(positions, graph.dynamic_input(0))};
            });
    };
    auto future_a = std::async(std::launch::async, [&] { return run(a, left, entered_a); });
    auto future_b = std::async(std::launch::async, [&] { return run(b, right, entered_b); });
    const bool overlap = ready_a.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
                         ready_b.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    release.set_value();
    const auto out_a = future_a.get(), out_b = future_b.get();
    require(overlap, "independent contexts serialize their graph builders");
    compare(out_a.at(0).values, rows(positions, left));
    compare(out_b.at(0).values, rows(positions, right));
    const auto after = model.weights->stats();
    require(after.host_tensors == before.host_tensors + 1, "concurrent cold initialization duplicated a weight");
    const auto other = b.stats();
    auto must_not_build = [](clap::detail::Graph &) -> std::vector<ggml_tensor *> {
        throw std::runtime_error("cache hit unexpectedly invoked its builder");
    };
    compare(a.run_cached("shared-context-barrier", {clap::detail::GraphInput(right)}, must_not_build).at(0).values,
            rows(positions, right));
    require(b.stats().builds == other.builds && b.stats().hits == other.hits, "one context modified another cache");
    try {
        a.run_cached("failed-context-build", {clap::detail::GraphInput(left)}, must_not_build);
        require(false, "failed builder unexpectedly succeeded");
    } catch (const std::runtime_error &error) {
        require(std::string(error.what()) == "cache hit unexpectedly invoked its builder", "wrong build failure");
    }
    compare(b.run_cached("shared-context-barrier", {clap::detail::GraphInput(left)}, must_not_build).at(0).values,
            rows(positions, left));
    require(b.stats().builds == other.builds, "a failed sibling evicted this context's graph");
    require(std::memcmp(table->data, snapshot.data(), snapshot.size()) == 0, "source weights were mutated");
}

void storage_variants() {
    for (const auto type :
         {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        for (const bool misaligned : {false, true}) {
            std::unique_ptr<ggml_context, decltype(&ggml_free)> metadata(ggml_init({65536, nullptr, true}), ggml_free);
            require(bool(metadata), "weight fixture allocation failed");
            auto *source = ggml_new_tensor_2d(metadata.get(), type, 256, 2);
            ggml_set_name(source, "shared.storage");
            std::vector<unsigned char> data(ggml_nbytes(source) + 128, 0);
            const auto aligned = (reinterpret_cast<uintptr_t>(data.data()) + 63) & ~uintptr_t(63);
            source->data = reinterpret_cast<void *>(aligned + (misaligned ? 1 : 0));
            const std::vector<unsigned char> snapshot(static_cast<unsigned char *>(source->data),
                                                      static_cast<unsigned char *>(source->data) + ggml_nbytes(source));
            clap::detail::ModelLoader loader(clap::RuntimeOptions{2, "cpu"});
            auto weights = std::make_shared<clap::detail::ModelWeights>(loader);
            clap::detail::GraphRuntime a(loader, 1, true, weights), b(loader, 2, true, weights);
            clap::detail::Graph one(a), two(b);
            auto *x = one.weight(source, false), *y = two.weight(source, false);
            require(x == y && x->buffer == y->buffer && x->type == type, "storage variant is not shared unchanged");
            const std::vector<int> ids{0, 1};
            compare(one.run(one.embedding(source, ids)), std::vector<float>(512, 0));
            compare(two.run(two.embedding(source, ids)), std::vector<float>(512, 0));
            require(weights->stats().host_tensors == 1, "aligned/copied storage was duplicated");
            require(std::memcmp(source->data, snapshot.data(), snapshot.size()) == 0, "storage bytes were modified");
        }
    }
}

struct DeviceEnvironment {
    bool present;
    std::string previous;
    DeviceEnvironment()
        : present(std::getenv("CLAP_DEVICE") != nullptr), previous(present ? std::getenv("CLAP_DEVICE") : "") {}
    static void set(const char *value) {
#ifdef _WIN32
        _putenv_s("CLAP_DEVICE", value ? value : "");
#else
        if (value)
            setenv("CLAP_DEVICE", value, 1);
        else
            unsetenv("CLAP_DEVICE");
#endif
    }
    ~DeviceEnvironment() { set(present ? previous.c_str() : nullptr); }
};
void frozen_options(const char *path, const nlohmann::json &fixture) {
    DeviceEnvironment environment;
    environment.set("cpu");
    auto options = clap_default_options();
    options.threads = 2;
    clap_model *raw = nullptr;
    check(clap_model_load(path, &options, &raw));
    Model model(raw, clap_model_free);
    environment.set("missing-device-after-model-load");
    auto defaults = clap_default_context_options();
    CLAP *null_raw = nullptr, *default_raw = nullptr;
    check(clap_context_create(model.get(), nullptr, &null_raw));
    Execution inherited(null_raw, clap_destroy);
    check(clap_context_create(model.get(), &defaults, &default_raw));
    Execution explicit_default(default_raw, clap_destroy);
    const auto &shared = inherited->context.model();
    require(inherited->context.runtime(shared.architecture.get(), shared.loader).stats().threads == 2 &&
                explicit_default->context.runtime(shared.architecture.get(), shared.loader).stats().threads == 2,
            "NULL and default context options did not inherit model threads");
    compare(encode(inherited.get(), fixture.at("cases").at(0).at("input")),
            fixture.at("cases").at(0).at("embedding").get<std::vector<float>>());
}
void context_oom(clap_model *model) {
#ifndef CLAP_SHARED
    CLAP *probe = nullptr;
    const auto begin = allocation_count.load();
    const auto status = clap_context_create(model, nullptr, &probe);
    const auto count = allocation_count.load() - begin;
    check(status);
    clap_destroy(probe);
    require(count > 3, "context allocation probe did not cover runtime construction");
    const auto owners = model->value.use_count();
    for (const auto budget : {0L, 1L, static_cast<long>(count - 2), static_cast<long>(count - 1)}) {
        CLAP *failed = reinterpret_cast<CLAP *>(1);
        allocation_budget.store(budget);
        const auto result = clap_context_create(model, nullptr, &failed);
        const bool diagnostic = std::strstr(clap_last_error(), "out of memory") != nullptr;
        allocation_budget.store(-1);
        require(result == CLAP_OUT_OF_MEMORY && !failed && diagnostic, "partial context OOM was not contained");
        require(model->value.use_count() == owners, "partial context leaked model ownership");
    }
#else
    (void)model;
#endif
}
void thread_errors(CLAP *first, CLAP *second, const std::string &text) {
    std::promise<void> failed, succeeded;
    auto first_failed = failed.get_future().share();
    auto other_succeeded = succeeded.get_future().share();
    auto a = std::async(std::launch::async, [&] {
        float *out = reinterpret_cast<float *>(1);
        size_t count = 9;
        const auto result = clap_encode_tokens(first, nullptr, 0, &out, &count);
        const std::string error = clap_last_error();
        failed.set_value();
        other_succeeded.wait();
        return result == CLAP_INVALID_ARGUMENT && !out && !count && !error.empty() && error == clap_last_error();
    });
    auto b = std::async(std::launch::async, [&] {
        first_failed.wait();
        try {
            encode(second, text);
            const bool empty = !*clap_last_error();
            succeeded.set_value();
            return empty;
        } catch (...) {
            succeeded.set_value();
            throw;
        }
    });
    require(a.get() && b.get(), "context calls mixed thread-local diagnostics");
}

void shared_oracle(const char *path, const char *case_path) {
    nlohmann::json spec;
    std::ifstream input(case_path);
    input >> spec;
    const auto mode = spec.at("mode").get<std::string>();
    const auto payload = spec.at("input").get<std::string>();
    clap::Audio audio;
    if (mode != "text")
        audio = clap::decode_audio(payload);
    auto options = clap_default_options();
    options.device = "cpu";
    options.threads = 2;
    clap_model *raw_model = nullptr;
    check(clap_model_load(path, &options, &raw_model));
    Model model(raw_model, clap_model_free);
    CLAP *raw_first = nullptr, *raw_second = nullptr;
    check(clap_context_create(model.get(), nullptr, &raw_first));
    Execution first(raw_first, clap_destroy);
    check(clap_context_create(model.get(), nullptr, &raw_second));
    Execution second(raw_second, clap_destroy);
    const auto &shared = first->context.model();
    require(&shared == &second->context.model(), "oracle contexts do not share a model");
    model.reset();
    auto run = [&](CLAP *context) -> nlohmann::json {
        if (mode == "text")
            return encode(context, payload);
        if (mode == "frames") {
            clap_audio_frames frames{};
            check(
                clap_get_audio_frames(context, audio.samples.data(), audio.samples.size(), audio.sample_rate, &frames));
            const std::unique_ptr<clap_audio_frames, decltype(&clap_free_audio_frames)> owned(&frames,
                                                                                              clap_free_audio_frames);
            return {{"frames", frames.frames},
                    {"dimension", frames.dimension},
                    {"values", std::vector<float>(frames.values, frames.values + frames.frames * frames.dimension)}};
        }
        require(mode == "audio", "unknown shared oracle mode");
        float *values = nullptr;
        size_t count = 0;
        check(clap_encode_audio(context, audio.samples.data(), audio.samples.size(), audio.sample_rate, 0, &values,
                                &count));
        Output owned(values, clap_free);
        return std::vector<float>(values, values + count);
    };
    require(run(first.get()) == spec.at("expected"), "first shared context differs from pinned baseline");
    const auto weights = shared.weights->stats();
    require(weights.host_tensors > 0, "oracle did not execute model weights");
    require(run(second.get()) == spec.at("expected"), "second shared context differs from pinned baseline");
    require(shared.weights->stats().host_tensors == weights.host_tensors &&
                shared.weights->stats().host_bytes == weights.host_bytes,
            "second oracle context duplicated model weights");
    first.reset();
    require(run(second.get()) == spec.at("expected"), "surviving shared context differs from pinned baseline");
    std::cout << "shared contexts match pinned baseline and retain one CPU weight bank\n";
}

int device_sharing() {
    const auto available = clap::available_devices();
    for (const auto &info : available) {
        if (!info.gpu)
            continue;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> metadata(ggml_init({65536, nullptr, true}), ggml_free);
        auto *source = ggml_new_tensor_2d(metadata.get(), GGML_TYPE_F32, 256, 65);
        ggml_set_name(source, "shared.device.weight");
        std::vector<float> values(ggml_nelements(source), 0.125f);
        source->data = values.data();
        clap::detail::ModelLoader loader(clap::RuntimeOptions{2, info.name, clap::ActivationPrecision::F32});
        auto weights = std::make_shared<clap::detail::ModelWeights>(loader);
        auto first = std::make_unique<clap::detail::GraphRuntime>(loader, 1, true, weights);
        auto second = std::make_unique<clap::detail::GraphRuntime>(loader, 2, true, weights);
        clap::detail::HostMatrix x(9, 256), y(9, 256);
        std::fill(x.data.begin(), x.data.end(), 0.25f);
        std::fill(y.data.begin(), y.data.end(), 0.5f);
        std::promise<void> entered_a, entered_b, release;
        auto ready_a = entered_a.get_future(), ready_b = entered_b.get_future();
        auto gate = release.get_future().share();
        auto run = [&](clap::detail::GraphRuntime &runtime, const clap::detail::HostMatrix &input,
                       std::promise<void> &entered) {
            return runtime.run_cached("device-sharing", {clap::detail::GraphInput(input)}, [&](clap::detail::Graph &g) {
                entered.set_value();
                gate.wait();
                return std::vector<ggml_tensor *>{
                    g.linear(g.dynamic_input(0), clap::detail::LayerWeights{source, nullptr})};
            });
        };
        auto a = std::async(std::launch::async, [&] { return run(*first, x, entered_a); });
        auto b = std::async(std::launch::async, [&] { return run(*second, y, entered_b); });
        const bool overlap = ready_a.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
                             ready_b.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
        release.set_value();
        const auto out_a = a.get(), out_b = b.get();
        require(overlap, "GPU runtimes share an execution lock");
        float expected_a = 0, expected_b = 0;
        for (int i = 0; i < 256; ++i) {
            expected_a += 0.125f * x[0][i];
            expected_b += 0.125f * y[0][i];
        }
        compare(out_a.at(0).values, std::vector<float>(9 * 65, expected_a));
        compare(out_b.at(0).values, std::vector<float>(9 * 65, expected_b));
        if (!first->stats().device_matmul || !second->stats().device_matmul)
            continue;
        require(weights->stats().uploads == 1 && weights->stats().device_tensors == 1, "GPU weight uploaded twice");
        require(first->stats().weight_uploads + second->stats().weight_uploads == 1,
                "per-context upload accounting differs");
        first.reset();
        compare(second
                    ->run_cached("device-sharing", {clap::detail::GraphInput(x)},
                                 [](clap::detail::Graph &) -> std::vector<ggml_tensor *> {
                                     throw std::runtime_error("surviving GPU context lost its graph");
                                 })
                    .at(0)
                    .values,
                std::vector<float>(9 * 65, expected_a));
        require(weights->stats().uploads == 1, "destroying allocating context discarded shared GPU weights");
        std::cout << "actual GPU matmul, concurrent contexts and one shared upload passed on " << info.name << '\n';
        return 0;
    }
    std::cout << "SKIP: no available GPU supports the sharing matmul fixture\n";
    return 77;
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--oracle") {
            shared_oracle(argv[2], argv[3]);
            return 0;
        }
        require(argc == 3 || (argc == 4 && std::string(argv[3]) == "--device"),
                "usage: test-context model.gguf fixture.json [--device]");
        if (argc == 4)
            return device_sharing();
        nlohmann::json fixture;
        std::ifstream input(argv[2]);
        input >> fixture;
        storage_variants();
        frozen_options(argv[1], fixture);
        auto options = clap_default_options();
        options.device = "cpu";
        options.threads = 2;
        clap_model *raw_model = nullptr;
        check(clap_model_load(argv[1], &options, &raw_model));
        Model model(raw_model, clap_model_free);
        auto first_options = clap_default_context_options();
        first_options.threads = 1;
        auto second_options = clap_default_context_options();
        second_options.threads = 2;
        CLAP *raw_first = nullptr, *raw_second = nullptr;
        check(clap_context_create(model.get(), &first_options, &raw_first));
        Execution first(raw_first, clap_destroy);
        check(clap_context_create(model.get(), &second_options, &raw_second));
        Execution second(raw_second, clap_destroy);
        require(first.get() != second.get(), "contexts must be distinct handles");
        require(clap_projection_dimension(first.get()) == fixture.at("dimensions"), "model properties lost");
        context_oom(model.get());
        graph_isolation(first.get(), second.get());
        CLAP *invalid = reinterpret_cast<CLAP *>(1);
        auto invalid_options = clap_default_context_options();
        invalid_options.threads = -1;
        require(clap_context_create(model.get(), &invalid_options, &invalid) == CLAP_INVALID_ARGUMENT && !invalid,
                "invalid context options must clear the output");
        invalid = reinterpret_cast<CLAP *>(1);
        require(clap_context_create(nullptr, nullptr, &invalid) == CLAP_INVALID_ARGUMENT && !invalid,
                "missing model must clear the context output");
        require(clap_context_create(model.get(), nullptr, nullptr) == CLAP_INVALID_ARGUMENT, "missing output accepted");
        model.reset();
        const auto &entry = fixture.at("cases").at(0);
        const auto text = entry.at("input").get<std::string>();
        thread_errors(first.get(), second.get(), text);
        auto work = [&](CLAP *context, size_t offset) {
            for (size_t pass = 0; pass < 3; ++pass)
                for (size_t i = 0; i < fixture.at("cases").size(); ++i) {
                    const auto &current = fixture.at("cases").at((i + offset) % fixture.at("cases").size());
                    compare(encode(context, current.at("input")), current.at("embedding").get<std::vector<float>>());
                }
        };
        auto a = std::async(std::launch::async, work, first.get(), 0);
        auto b = std::async(std::launch::async, work, second.get(), 1);
        a.get();
        b.get();
        float *retained = nullptr;
        size_t retained_size = 0;
        check(clap_encode_text(first.get(), text.c_str(), &retained, &retained_size));
        Output output(retained, clap_free);
        first.reset();
        compare(encode(second.get(), text), entry.at("embedding").get<std::vector<float>>());
        second.reset();
        compare({retained, retained + retained_size}, entry.at("embedding").get<std::vector<float>>());
        clap_model_free(nullptr);
        std::cout << "shared-model contexts, cold buffer identity, isolated caches, overlap, OOM, scalar oracles and "
                     "retained outputs passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
