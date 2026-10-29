#include "clap.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

void validate_samples(const std::vector<float> &input) {
    if (input.empty()) {
        throw std::invalid_argument("PCM must be a nonempty mono sequence");
    }
    for (float value : input) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("PCM samples must be finite");
        }
    }
}

void validate_rate(int rate) {
    if (rate <= 0) throw std::invalid_argument("sample rate must be positive");
}

std::vector<std::vector<float>> matrix(const clap::AudioFrames &frames) {
    std::vector<std::vector<float>> output;
    output.reserve(frames.frames);
    for (int i = 0; i < frames.frames; ++i) {
        const auto begin = frames.values.begin() + i * frames.dimension;
        output.emplace_back(begin, begin + frames.dimension);
    }
    return output;
}

// Always release the GIL BEFORE taking the model mutex; release the mutex
// BEFORE reacquiring the GIL. close(), properties and inference share this order.
class CLAP {
public:
    CLAP(const std::filesystem::path &path, int threads) {
        if (threads < 0) throw std::invalid_argument("threads must be nonnegative");
        py::gil_scoped_release release;
        model_ = std::make_unique<clap::CLAP>(path.u8string(), threads);
    }

    template <typename F>
    auto call(F &&function) {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!model_) throw std::runtime_error("CLAP is closed");
        return function(*model_);
    }

    void close() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        model_.reset();
    }

private:
    std::unique_ptr<clap::CLAP> model_;
    std::mutex mutex_;
};

} // namespace

PYBIND11_MODULE(_clap_cpp, module) {
    module.doc() = "Native CPU inference for clap.cpp";
    py::class_<CLAP>(module, "CLAP")
        .def(py::init<const std::filesystem::path &, int>(), py::arg("path"), py::arg("threads") = 0)
        .def_property_readonly("projection_dimension", [](CLAP &self) {
            return self.call([](clap::CLAP &model) { return model.projection_dimension(); });
        })
        .def_property_readonly("audio_sample_rate", [](CLAP &self) {
            return self.call([](clap::CLAP &model) { return model.audio_sample_rate(); });
        })
        .def("tokenize", [](CLAP &self, const std::string &text, py::ssize_t max_length) {
            if (max_length <= 0) throw std::invalid_argument("max_length must be positive");
            return self.call([&](clap::CLAP &model) { return model.tokenize(text, max_length); });
        }, py::arg("text"), py::arg("max_length") = 512)
        .def("encode_text", [](CLAP &self, const std::string &text) {
            return self.call([&](clap::CLAP &model) { return model.encode_text(text); });
        }, py::arg("text"))
        .def("encode_text", [](CLAP &self, const std::vector<int> &tokens) {
            return self.call([&](clap::CLAP &model) { return model.encode_text(tokens); });
        }, py::arg("tokens"))
        .def("encode_audio", [](CLAP &self, const std::vector<float> &pcm, int sample_rate, py::ssize_t crop_offset) {
            validate_rate(sample_rate);
            if (crop_offset < 0) throw std::invalid_argument("crop_offset must be nonnegative");
            validate_samples(pcm);
            return self.call([&](clap::CLAP &model) { return model.encode_audio(pcm, sample_rate, crop_offset); });
        }, py::arg("pcm"), py::arg("sample_rate") = 48000, py::arg("crop_offset") = 0)
        .def("encode_audio", [](CLAP &self, const std::filesystem::path &path, py::ssize_t crop_offset) {
            if (crop_offset < 0) throw std::invalid_argument("crop_offset must be nonnegative");
            const auto filename = path.u8string();
            return self.call([&](clap::CLAP &model) {
                const auto decoded = clap::decode_audio(filename);
                return model.encode_audio(decoded.samples, decoded.sample_rate, crop_offset);
            });
        }, py::arg("path"), py::arg("crop_offset") = 0)
        .def("audio_frames", [](CLAP &self, const std::vector<float> &pcm, int sample_rate) {
            validate_rate(sample_rate);
            validate_samples(pcm);
            return matrix(self.call([&](clap::CLAP &model) { return model.audio_frames(pcm, sample_rate); }));
        }, py::arg("pcm"), py::arg("sample_rate") = 48000)
        .def("clamp3_features", [](CLAP &self, const std::vector<float> &pcm, int sample_rate) {
            validate_rate(sample_rate);
            validate_samples(pcm);
            return matrix(self.call([&](clap::CLAP &model) { return model.clamp3_features(pcm, sample_rate); }));
        }, py::arg("pcm"), py::arg("sample_rate") = 24000)
        .def("close", &CLAP::close)
        .def("__enter__", [](CLAP &self) -> CLAP & {
            self.call([](clap::CLAP &) {});
            return self;
        }, py::return_value_policy::reference_internal)
        .def("__exit__", [](CLAP &self, py::object, py::object, py::object) { self.close(); });

}
