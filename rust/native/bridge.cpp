// Private ABI for the Rust crate, not a public clap.cpp C API.
#include "clap.h"
#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Result {
    std::unique_ptr<CLAP, decltype(&clap_destroy)> model{nullptr, clap_destroy};
    std::vector<float> floats;
    char error[1024] = {};
};
void check(clap_status status) {
    if (status == CLAP_OK)
        return;
    const std::string message = clap_last_error();
    if (status == CLAP_INVALID_ARGUMENT)
        throw std::invalid_argument(message);
    if (status == CLAP_OUT_OF_MEMORY)
        throw std::bad_alloc();
    throw std::runtime_error(message);
}

template <typename T> std::vector<T> take_buffer(T *data, std::size_t count) {
    const std::unique_ptr<T, decltype(&clap_free)> owned(data, clap_free);
    if (!count)
        return {};
    return {data, data + count};
}

} // namespace
extern "C" {
// Arguments are borrowed synchronously. cr_free releases every result.
// Inline diagnostics keep exception translation allocation-free.
// Ops: 0=open, 1=encode_text, 2=encode_audio (file).
Result *cr_call(int op, CLAP *model, const char *s, int threads, size_t crop_offset) noexcept {
    auto result = std::unique_ptr<Result>(new (std::nothrow) Result);
    if (!result)
        return nullptr;
    try {
        switch (op) {
        case 0:
            if (threads < 0)
                throw std::invalid_argument("threads must be nonnegative");
            {
                auto options = clap_default_options();
                options.threads = threads;
                CLAP *created = nullptr;
                check(clap_create(s, &options, &created));
                result->model.reset(created);
            }
            break;
        case 1: {
            float *values = nullptr;
            std::size_t count = 0;
            check(clap_encode_text(model, s, &values, &count));
            result->floats = take_buffer(values, count);
        } break;
        case 2: {
            float *values = nullptr;
            std::size_t count = 0;
            check(clap_encode_audio_file(model, s, crop_offset, &values, &count));
            result->floats = take_buffer(values, count);
            break;
        }
        default:
            throw std::invalid_argument("unknown operation");
        }
    } catch (const std::exception &e) {
        std::snprintf(result->error, sizeof(result->error), "%s", e.what());
    } catch (...) {
        std::snprintf(result->error, sizeof(result->error), "%s", "unknown native exception");
    }
    return result.release();
}
void cr_free(Result *p) noexcept { delete p; }
void cr_model_free(CLAP *p) noexcept { clap_destroy(p); }
CLAP *cr_model_take(Result *p) noexcept { return p->model.release(); }
const char *cr_error(const Result *p) noexcept { return p->error; }
const float *cr_floats(const Result *p) noexcept { return p->floats.data(); }
size_t cr_len(const Result *p) noexcept { return p->floats.size(); }
}
