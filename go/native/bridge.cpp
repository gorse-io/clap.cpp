#include "bridge.h"
#include "clap.h"
#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

struct cg_model {
    clap::CLAP model;
    cg_model(const char *path, int threads) : model(path, threads) {}
};
struct cg_result {
    std::unique_ptr<cg_model> model;
    std::vector<float> values;
    char error[1024] = {};
};
namespace {
template<class F> cg_result *run(F f) noexcept {
    std::unique_ptr<cg_result> result(new (std::nothrow) cg_result);
    if (!result) return nullptr;
    try { f(*result); }
    catch (const std::exception &e) { std::snprintf(result->error, sizeof(result->error), "%s", e.what()); }
    catch (...) { std::snprintf(result->error, sizeof(result->error), "%s", "unknown native exception"); }
    return result.release();
}
}
extern "C" {
cg_result *cg_open(const char *path, int threads) noexcept {
    return run([&](cg_result &r) {
        if (!path || threads < 0) throw std::invalid_argument("invalid model arguments");
        r.model = std::make_unique<cg_model>(path, threads);
    });
}
cg_result *cg_text(cg_model *m, const char *text) noexcept {
    return run([&](cg_result &r) {
        if (!m || !text) throw std::invalid_argument("invalid text arguments");
        r.values = m->model.encode_text(text);
    });
}
cg_result *cg_audio(cg_model *m, const char *path, size_t crop) noexcept {
    return run([&](cg_result &r) {
        if (!m || !path) throw std::invalid_argument("invalid audio arguments");
        const auto audio = clap::decode_audio(path);
        r.values = m->model.encode_audio(audio.samples, audio.sample_rate, crop);
    });
}
const char *cg_error(const cg_result *r) noexcept { return r->error; }
cg_model *cg_take_model(cg_result *r) noexcept { return r->model.release(); }
const float *cg_data(const cg_result *r) noexcept { return r->values.data(); }
size_t cg_size(const cg_result *r) noexcept { return r->values.size(); }
void cg_result_free(cg_result *r) noexcept { delete r; }
void cg_model_free(cg_model *m) noexcept { delete m; }
}
