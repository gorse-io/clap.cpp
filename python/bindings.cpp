#include "clap.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using PCM = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

std::vector<float> samples(const PCM &input) {
    if (input.ndim() != 1 || input.size() == 0) {
        throw std::invalid_argument("PCM must be a nonempty one-dimensional mono array");
    }
    std::vector<float> result(input.data(), input.data() + input.size());
    for (float value : result) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("PCM samples must be finite");
        }
    }
    return result;
}

void validate_rate(int rate) {
    if (rate <= 0) throw std::invalid_argument("sample rate must be positive");
}

py::array_t<float> array(const std::vector<float> &values) {
    py::array_t<float> output(values.size());
    if (!values.empty()) std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(float));
    return output;
}

py::array_t<float> matrix(const clap::AudioFrames &frames) {
    py::array_t<float> output({frames.frames, frames.dimension});
    if (!frames.values.empty()) {
        std::memcpy(output.mutable_data(), frames.values.data(), frames.values.size() * sizeof(float));
    }
    return output;
}

// Always release the GIL BEFORE taking the model mutex; release the mutex
// BEFORE reacquiring the GIL. close(), properties and inference share this order.
class Model {
public:
    Model(const std::filesystem::path &path, int threads) {
        if (threads < 0) throw std::invalid_argument("threads must be nonnegative");
        py::gil_scoped_release release;
        model_ = std::make_unique<clap::Model>(path.u8string(), threads);
    }

    template <typename F>
    auto call(F &&function) {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!model_) throw std::runtime_error("Model is closed");
        return function(*model_);
    }

    void close() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        model_.reset();
    }

private:
    std::unique_ptr<clap::Model> model_;
    std::mutex mutex_;
};

} // namespace

PYBIND11_MODULE(_clap_cpp, module) {
    module.doc() = "Native inference for clap.cpp";
    py::class_<Model>(module, "Model")
        .def(py::init<const std::filesystem::path &, int>(), py::arg("path"), py::arg("threads") = 0)
        .def_property_readonly("projection_dimension", [](Model &self) {
            return self.call([](clap::Model &model) { return model.projection_dimension(); });
        })
        .def_property_readonly("audio_sample_rate", [](Model &self) {
            return self.call([](clap::Model &model) { return model.audio_sample_rate(); });
        })
        .def("tokenize", [](Model &self, const std::string &text, py::ssize_t max_length) {
            if (max_length <= 0) throw std::invalid_argument("max_length must be positive");
            return self.call([&](clap::Model &model) { return model.tokenize(text, max_length); });
        }, py::arg("text"), py::arg("max_length") = 512)
        .def("text", [](Model &self, const std::string &text) {
            return array(self.call([&](clap::Model &model) { return model.text(text); }));
        }, py::arg("text"))
        .def("text", [](Model &self, const std::vector<int> &tokens) {
            return array(self.call([&](clap::Model &model) { return model.text(tokens); }));
        }, py::arg("tokens"))
        .def("audio", [](Model &self, const PCM &pcm, int sample_rate, py::ssize_t crop_offset) {
            validate_rate(sample_rate);
            if (crop_offset < 0) throw std::invalid_argument("crop_offset must be nonnegative");
            auto input = samples(pcm);
            return array(self.call([&](clap::Model &model) { return model.audio(input, sample_rate, crop_offset); }));
        }, py::arg("pcm"), py::arg("sample_rate") = 48000, py::arg("crop_offset") = 0)
        .def("audio_frames", [](Model &self, const PCM &pcm, int sample_rate) {
            validate_rate(sample_rate);
            auto input = samples(pcm);
            return matrix(self.call([&](clap::Model &model) { return model.audio_frames(input, sample_rate); }));
        }, py::arg("pcm"), py::arg("sample_rate") = 48000)
        .def("clamp3_features", [](Model &self, const PCM &pcm, int sample_rate) {
            validate_rate(sample_rate);
            auto input = samples(pcm);
            return matrix(self.call([&](clap::Model &model) { return model.clamp3_features(input, sample_rate); }));
        }, py::arg("pcm"), py::arg("sample_rate") = 24000)
        .def("close", &Model::close)
        .def("__enter__", [](Model &self) -> Model & {
            self.call([](clap::Model &) {});
            return self;
        }, py::return_value_policy::reference_internal)
        .def("__exit__", [](Model &self, py::object, py::object, py::object) { self.close(); });

    module.def("decode_audio", [](const std::filesystem::path &path) {
        clap::Audio decoded;
        {
            py::gil_scoped_release release;
            decoded = clap::decode_audio(path.u8string());
        }
        return py::make_tuple(array(decoded.samples), decoded.sample_rate);
    }, py::arg("path"));
    module.def("resample_audio", [](const PCM &pcm, int source_rate, int target_rate) {
        validate_rate(source_rate);
        validate_rate(target_rate);
        clap::Audio input{samples(pcm), source_rate};
        clap::Audio output;
        {
            py::gil_scoped_release release;
            output = clap::resample_audio(input, target_rate);
        }
        return array(output.samples);
    }, py::arg("pcm"), py::arg("source_rate"), py::arg("target_rate") = 48000);
}
