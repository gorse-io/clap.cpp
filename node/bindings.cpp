#include <napi.h>
#include "clap.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// No C++ exception may escape a Node-API callback, including allocation errors.
template <class F>
Napi::Value guarded(Napi::Env env, F &&fn) {
    try { return fn(); }
    catch (const Napi::Error &e) { e.ThrowAsJavaScriptException(); }
    catch (const std::invalid_argument &e) { Napi::TypeError::New(env, e.what()).ThrowAsJavaScriptException(); }
    catch (const std::exception &e) { Napi::Error::New(env, e.what()).ThrowAsJavaScriptException(); }
    catch (...) { Napi::Error::New(env, "Unknown native exception").ThrowAsJavaScriptException(); }
    return env.Undefined();
}

std::string string(const Napi::Value &v, const char *name) {
    if (!v.IsString()) throw std::invalid_argument(std::string(name) + " must be a string");
    return v.As<Napi::String>().Utf8Value();
}
std::string path(const Napi::Value &v) {
    auto p = string(v, "path");
    if (p.empty() || p.find('\0') != std::string::npos)
        throw std::invalid_argument("path must be nonempty and contain no NUL bytes");
    return p;
}
std::size_t integer(const Napi::Value &v, const char *name, double min, double max) {
    if (!v.IsNumber()) throw std::invalid_argument(std::string(name) + " must be an integer");
    const double n = v.As<Napi::Number>().DoubleValue();
    if (!std::isfinite(n) || std::floor(n) != n || n < min || n > max)
        throw std::invalid_argument(std::string(name) + " is outside the supported integer range");
    return static_cast<std::size_t>(n);
}
int rate(const Napi::Value &v) {
    return static_cast<int>(integer(v, "sampleRate", 1, std::numeric_limits<int>::max()));
}
int optional_rate(const Napi::Value &v, int fallback) { return v.IsUndefined() ? fallback : rate(v); }
std::vector<float> samples(const Napi::Value &v) {
    if (!v.IsTypedArray() || v.As<Napi::TypedArray>().TypedArrayType() != napi_float32_array)
        throw std::invalid_argument("PCM must be a nonempty Float32Array of mono samples");
    auto a = v.As<Napi::Float32Array>();
    if (a.ElementLength() == 0 || a.ElementLength() > (std::size_t(1) << 28) || !a.Data())
        throw std::invalid_argument("PCM must be nonempty and at most 1 GiB");
    std::vector<float> result(a.Data(), a.Data() + a.ElementLength());
    for (float x : result) if (!std::isfinite(x)) throw std::invalid_argument("PCM samples must be finite");
    return result;
}
std::vector<int> tokens(const Napi::Value &v) {
    std::vector<int> result;
    if (v.IsTypedArray() && v.As<Napi::TypedArray>().TypedArrayType() == napi_int32_array) {
        auto a = v.As<Napi::Int32Array>();
        if (a.ElementLength() && a.Data()) result.assign(a.Data(), a.Data() + a.ElementLength());
        for (int id : result) if (id < 0) throw std::invalid_argument("token IDs must be nonnegative");
    } else if (v.IsArray()) {
        auto a = v.As<Napi::Array>();
        result.reserve(a.Length());
        for (uint32_t i = 0; i < a.Length(); ++i)
            result.push_back(static_cast<int>(integer(a.Get(i), "token ID", 0, std::numeric_limits<int>::max())));
    } else { throw std::invalid_argument("text must be a string, Int32Array or number[]"); }
    if (result.empty()) throw std::invalid_argument("tokens must be nonempty");
    return result;
}
Napi::Float32Array array(Napi::Env env, const std::vector<float> &v) {
    auto output = Napi::Float32Array::New(env, v.size());
    if (!v.empty()) std::memcpy(output.Data(), v.data(), v.size() * sizeof(float));
    return output;
}
class CLAP : public Napi::ObjectWrap<CLAP> {
public:
    static Napi::Function define(Napi::Env env) {
        return DefineClass(env, "CLAP", {
            InstanceMethod<&CLAP::encode_text>("encodeText"),
            InstanceMethod<&CLAP::encode_audio>("encodeAudio"),
            InstanceMethod<&CLAP::close>("close")
        });
    }
    explicit CLAP(const Napi::CallbackInfo &info) : Napi::ObjectWrap<CLAP>(info) {
        guarded(info.Env(), [&]() -> Napi::Value {
            auto p = path(info[0]);
            int threads = info[1].IsUndefined() ? 0 : static_cast<int>(integer(info[1], "threads", 0, std::numeric_limits<int>::max()));
            model_ = std::make_unique<clap::CLAP>(p, threads);
            return info.Env().Undefined();
        });
    }
private:
    // Synchronous calls cannot overlap within one JS isolate; each worker owns its
    // own ObjectWrap. There is no globally shared model handle or borrowed output.
    std::unique_ptr<clap::CLAP> model_;
    clap::CLAP &get() {
        if (!model_) throw std::runtime_error("CLAP is closed");
        return *model_;
    }
    Napi::Value encode_text(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value {
            // Array element getters can reenter JS and close this model. Parse
            // completely before taking a native reference, then recheck ownership.
            if (info[0].IsString()) {
                auto s = string(info[0], "text");
                return array(info.Env(), get().encode_text(s));
            }
            auto ids = tokens(info[0]);
            return array(info.Env(), get().encode_text(ids));
        });
    }
    Napi::Value encode_audio(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value {
            if (info[0].IsString()) {
                const auto filename = path(info[0]);
                const auto crop = info[1].IsUndefined() ? 0 : integer(info[1], "cropOffset", 0,
                    std::min(9007199254740991.0, static_cast<double>(std::numeric_limits<std::size_t>::max())));
                auto &m = get();
                auto decoded = clap::decode_audio(filename);
                return array(info.Env(), m.encode_audio(decoded.samples, decoded.sample_rate, crop));
            }
            auto pcm = samples(info[0]);
            int sr = optional_rate(info[1], 48000);
            auto crop = info[2].IsUndefined() ? 0 : integer(info[2], "cropOffset", 0,
                std::min(9007199254740991.0, static_cast<double>(std::numeric_limits<std::size_t>::max())));
            return array(info.Env(), get().encode_audio(pcm, sr, crop));
        });
    }
    Napi::Value close(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value { model_.reset(); return info.Env().Undefined(); });
    }
};

Napi::Object initialize(Napi::Env env, Napi::Object exports) {
    exports.Set("CLAP", CLAP::define(env));
    return exports;
}
} // namespace
NODE_API_MODULE(clap_node, initialize)
