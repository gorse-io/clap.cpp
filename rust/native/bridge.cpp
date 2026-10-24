// Private ABI for the Rust crate, not a public clap.cpp C API.
#include "clap.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
struct Result {
    std::unique_ptr<clap::Model> model;
    std::vector<float> floats;
    std::vector<int> ints;
    int rows = 0, cols = 0, rate = 0;
    char error[1024] = {};
};
static_assert(sizeof(int) == sizeof(int32_t), "Rust tokens require 32-bit int");
std::vector<float> pcm(const float *p, size_t n, int rate) {
    if (rate <= 0 || n == 0 || n > (size_t(1) << 28))
        throw std::invalid_argument("PCM must be nonempty, <=1 GiB, with positive sample rate");
    std::vector<float> v(p, p + n);
    for (float x : v) if (!std::isfinite(x)) throw std::invalid_argument("PCM must be finite");
    return v;
}
}
extern "C" {
// All arguments are borrowed for this call. Returned Result owns all outputs;
// cr_free must release it. On allocation failure returns null. Errors use an
// inline buffer, so translating bad_alloc cannot itself allocate or throw.
// Ops: 0=open, 1=tokenize, 2=text, 3=tokens, 4=audio, 5=frames,
// 6=clamp3_features, 7=metadata, 8=decode, 9=resample, 10=mel.
Result *cr_call(int op, clap::Model *model, const char *s, const float *f,
                const int *ids, size_t n, int rate, size_t extra) noexcept {
    auto result = std::unique_ptr<Result>(new (std::nothrow) Result);
    if (!result) return nullptr;
    try {
        switch (op) {
        case 0:
            if (rate < 0) throw std::invalid_argument("threads must be nonnegative");
            result->model = std::make_unique<clap::Model>(s, rate); break;
        case 1:
            if (!extra) throw std::invalid_argument("max_length must be positive");
            result->ints = model->tokenize(s, extra); break;
        case 2: result->floats = model->text(s); break;
        case 3:
            if (!n) throw std::invalid_argument("tokens must be nonempty");
            result->floats = model->text(std::vector<int>(ids, ids + n)); break;
        case 4: result->floats = model->audio(pcm(f, n, rate), rate, extra); break;
        case 5: case 6: {
            auto input = pcm(f, n, rate);
            auto frames = op == 5 ? model->audio_frames(input, rate) : model->clamp3_features(input, rate);
            result->rows = frames.frames; result->cols = frames.dimension;
            result->floats = std::move(frames.values); break;
        }
        case 7: result->rate = model->audio_sample_rate(); result->cols = model->projection_dimension(); break;
        case 8: {
            auto a = clap::decode_audio(s);
            result->floats = std::move(a.samples); result->rate = a.sample_rate; break;
        }
        case 9: {
            if (extra > size_t(INT32_MAX)) throw std::invalid_argument("target rate out of range");
            auto a = clap::resample_audio({pcm(f, n, rate), rate}, int(extra));
            result->floats = std::move(a.samples); break;
        }
        case 10: result->floats = clap::log_mel(pcm(f, n, rate), rate, extra); break;
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
void cr_model_free(clap::Model *p) noexcept { delete p; }
clap::Model *cr_model_take(Result *p) noexcept { return p->model.release(); }
const char *cr_error(const Result *p) noexcept { return p->error; }
const float *cr_floats(const Result *p) noexcept { return p->floats.data(); }
const int *cr_ints(const Result *p) noexcept { return p->ints.data(); }
size_t cr_len(const Result *p, int ints) noexcept { return ints ? p->ints.size() : p->floats.size(); }
int cr_meta(const Result *p, int field) noexcept { return field == 0 ? p->rows : field == 1 ? p->cols : p->rate; }
}
