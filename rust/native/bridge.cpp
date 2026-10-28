// Private ABI for the Rust crate, not a public clap.cpp C API.
#include "clap.h"
#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
struct Result {
    std::unique_ptr<clap::CLAP> model;
    std::vector<float> floats;
    char error[1024] = {};
};
}
extern "C" {
// Arguments are borrowed synchronously. cr_free releases every result.
// Inline diagnostics keep exception translation allocation-free.
// Ops: 0=open, 1=encode_text, 2=encode_audio (file).
Result *cr_call(int op, clap::CLAP *model, const char *s,
                int threads, size_t crop_offset) noexcept {
    auto result = std::unique_ptr<Result>(new (std::nothrow) Result);
    if (!result) return nullptr;
    try {
        switch (op) {
        case 0:
            if (threads < 0) throw std::invalid_argument("threads must be nonnegative");
            result->model = std::make_unique<clap::CLAP>(s, threads); break;
        case 1: result->floats = model->encode_text(s); break;
        case 2: {
            const auto audio = clap::decode_audio(s);
            result->floats = model->encode_audio(audio.samples, audio.sample_rate, crop_offset);
            break;
        }
        default: throw std::invalid_argument("unknown operation");
        }
    } catch (const std::exception &e) {
        std::snprintf(result->error, sizeof(result->error), "%s", e.what());
    } catch (...) {
        std::snprintf(result->error, sizeof(result->error), "%s", "unknown native exception");
    }
    return result.release();
}
void cr_free(Result *p) noexcept { delete p; }
void cr_model_free(clap::CLAP *p) noexcept { delete p; }
clap::CLAP *cr_model_take(Result *p) noexcept { return p->model.release(); }
const char *cr_error(const Result *p) noexcept { return p->error; }
const float *cr_floats(const Result *p) noexcept { return p->floats.data(); }
size_t cr_len(const Result *p) noexcept { return p->floats.size(); }
}
