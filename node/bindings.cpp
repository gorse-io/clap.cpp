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
Napi::Object frames(Napi::Env env, const clap::AudioFrames &f) {
    auto output = Napi::Object::New(env);
    output.Set("values", array(env, f.values));
    output.Set("frames", Napi::Number::New(env, f.frames));
    output.Set("dimension", Napi::Number::New(env, f.dimension));
    return output;
}

class Model : public Napi::ObjectWrap<Model> {
public:
    static Napi::Function define(Napi::Env env) {
        return DefineClass(env, "Model", {
            InstanceAccessor<&Model::projection_dimension>("projectionDimension"),
            InstanceAccessor<&Model::audio_sample_rate>("audioSampleRate"),
            InstanceMethod<&Model::tokenize>("tokenize"),
            InstanceMethod<&Model::text>("text"),
            InstanceMethod<&Model::audio>("audio"),
            InstanceMethod<&Model::audio_frames>("audioFrames"),
            InstanceMethod<&Model::clamp3_features>("clamp3Features"),
            InstanceMethod<&Model::close>("close")
        });
    }
    explicit Model(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Model>(info) {
        guarded(info.Env(), [&]() -> Napi::Value {
            auto p = path(info[0]);
            int threads = info[1].IsUndefined() ? 0 : static_cast<int>(integer(info[1], "threads", 0, std::numeric_limits<int>::max()));
            model_ = std::make_unique<clap::Model>(p, threads);
            return info.Env().Undefined();
        });
    }
private:
    // Synchronous calls cannot overlap within one JS isolate; each worker owns its
    // own ObjectWrap. There is no globally shared model handle or borrowed output.
    std::unique_ptr<clap::Model> model_;
    clap::Model &get() {
        if (!model_) throw std::runtime_error("Model is closed");
        return *model_;
    }
    Napi::Value projection_dimension(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value { return Napi::Number::New(info.Env(), get().projection_dimension()); });
    }
    Napi::Value audio_sample_rate(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value { return Napi::Number::New(info.Env(), get().audio_sample_rate()); });
    }
    Napi::Value tokenize(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value {
            auto &m = get();
            auto s = string(info[0], "text");
            auto n = info[1].IsUndefined() ? 512 : integer(info[1], "maxLength", 1, std::numeric_limits<int>::max());
            auto ids = m.tokenize(s, n);
            auto out = Napi::Int32Array::New(info.Env(), ids.size());
            for (std::size_t i = 0; i < ids.size(); ++i) out[i] = ids[i];
            return out;
        });
    }
    Napi::Value text(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value {
            // Array element getters can reenter JS and close this model. Parse
            // completely before taking a native reference, then recheck ownership.
            if (info[0].IsString()) {
                auto s = string(info[0], "text");
                return array(info.Env(), get().text(s));
            }
            auto ids = tokens(info[0]);
            return array(info.Env(), get().text(ids));
        });
    }
    Napi::Value audio(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value {
            auto &m = get();
            auto pcm = samples(info[0]);
            int sr = optional_rate(info[1], 48000);
            auto crop = info[2].IsUndefined() ? 0 : integer(info[2], "cropOffset", 0,
                std::min(9007199254740991.0, static_cast<double>(std::numeric_limits<std::size_t>::max())));
            return array(info.Env(), m.audio(pcm, sr, crop));
        });
    }
    Napi::Value audio_frames(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value {
            auto &m = get();
            auto pcm = samples(info[0]);
            return frames(info.Env(), m.audio_frames(pcm, optional_rate(info[1], 48000)));
        });
    }
    Napi::Value clamp3_features(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value {
            auto &m = get();
            auto pcm = samples(info[0]);
            return frames(info.Env(), m.clamp3_features(pcm, optional_rate(info[1], 24000)));
        });
    }
    Napi::Value close(const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value { model_.reset(); return info.Env().Undefined(); });
    }
};

Napi::Object initialize(Napi::Env env, Napi::Object exports) {
    exports.Set("Model", Model::define(env));
    exports.Set("decodeAudio", Napi::Function::New(env, [](const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value {
            auto decoded = clap::decode_audio(path(info[0]));
            auto result = Napi::Object::New(info.Env());
            result.Set("samples", array(info.Env(), decoded.samples));
            result.Set("sampleRate", Napi::Number::New(info.Env(), decoded.sample_rate));
            return result;
        });
    }, "decodeAudio"));
    exports.Set("resampleAudio", Napi::Function::New(env, [](const Napi::CallbackInfo &info) {
        return guarded(info.Env(), [&]() -> Napi::Value {
            clap::Audio input{samples(info[0]), rate(info[1])};
            return array(info.Env(), clap::resample_audio(input, optional_rate(info[2], 48000)).samples);
        });
    }, "resampleAudio"));
    return exports;
}
} // namespace
NODE_API_MODULE(clap_node, initialize)
