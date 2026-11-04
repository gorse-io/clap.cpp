#include "clap.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

std::string utf8_path(const std::filesystem::path &path) {
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

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
    if (rate <= 0)
        throw std::invalid_argument("sample rate must be positive");
}

const char *c_string(const std::string &value) {
    if (value.find('\0') != std::string::npos)
        throw std::invalid_argument("text and paths must contain no NUL bytes");
    return value.c_str();
}

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

// Always release the GIL BEFORE taking the model mutex; release the mutex
// BEFORE reacquiring the GIL. close(), properties and inference share this order.
class PythonCLAP {
  public:
    PythonCLAP(const std::filesystem::path &path, int threads) {
        if (threads < 0)
            throw std::invalid_argument("threads must be nonnegative");
        py::gil_scoped_release release;
        const auto filename = utf8_path(path);
        auto options = clap_default_options();
        options.threads = threads;
        ::CLAP *model = nullptr;
        check(clap_create(c_string(filename), &options, &model));
        model_.reset(model);
    }

    template <typename F> auto call(F &&function) {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!model_)
            throw std::runtime_error("CLAP is closed");
        return function(model_.get());
    }

    void close() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        model_.reset();
    }

  private:
    std::unique_ptr<::CLAP, decltype(&clap_destroy)> model_{nullptr, clap_destroy};
    std::mutex mutex_;
};

} // namespace

PYBIND11_MODULE(_clap_cpp, module) {
    module.doc() = "Native inference for clap.cpp";
    py::class_<PythonCLAP>(module, "CLAP")
        .def(py::init<const std::filesystem::path &, int>(), py::arg("path"), py::arg("threads") = 0)
        .def_property_readonly("projection_dimension",
                               [](PythonCLAP &self) {
                                   return self.call([](::CLAP *model) {
                                       int value = 0;
                                       check(clap_projection_dimension(model, &value));
                                       return value;
                                   });
                               })
        .def_property_readonly("audio_sample_rate",
                               [](PythonCLAP &self) {
                                   return self.call([](::CLAP *model) {
                                       int value = 0;
                                       check(clap_audio_sample_rate(model, &value));
                                       return value;
                                   });
                               })
        .def(
            "tokenize",
            [](PythonCLAP &self, const std::string &text, py::ssize_t max_length) {
                if (max_length <= 0)
                    throw std::invalid_argument("max_length must be positive");
                return self.call([&](::CLAP *model) {
                    int *values = nullptr;
                    std::size_t count = 0;
                    check(clap_tokenize(model, c_string(text), max_length, &values, &count));
                    return take_buffer(values, count);
                });
            },
            py::arg("text"), py::arg("max_length") = 512)
        .def(
            "encode_text",
            [](PythonCLAP &self, const std::string &text) {
                return self.call([&](::CLAP *model) {
                    float *values = nullptr;
                    std::size_t count = 0;
                    check(clap_encode_text(model, c_string(text), &values, &count));
                    return take_buffer(values, count);
                });
            },
            py::arg("text"))
        .def(
            "encode_text",
            [](PythonCLAP &self, const std::vector<int> &tokens) {
                return self.call([&](::CLAP *model) {
                    float *values = nullptr;
                    std::size_t count = 0;
                    check(clap_encode_tokens(model, tokens.data(), tokens.size(), &values, &count));
                    return take_buffer(values, count);
                });
            },
            py::arg("tokens"))
        .def(
            "encode_audio",
            [](PythonCLAP &self, const std::vector<float> &pcm, int sample_rate, py::ssize_t crop_offset) {
                validate_rate(sample_rate);
                if (crop_offset < 0)
                    throw std::invalid_argument("crop_offset must be nonnegative");
                validate_samples(pcm);
                return self.call([&](::CLAP *model) {
                    float *values = nullptr;
                    std::size_t count = 0;
                    check(clap_encode_audio(model, pcm.data(), pcm.size(), sample_rate, crop_offset, &values, &count));
                    return take_buffer(values, count);
                });
            },
            py::arg("pcm"), py::arg("sample_rate") = 48000, py::arg("crop_offset") = 0)
        .def(
            "encode_audio",
            [](PythonCLAP &self, const std::filesystem::path &path, py::ssize_t crop_offset) {
                if (crop_offset < 0)
                    throw std::invalid_argument("crop_offset must be nonnegative");
                const auto filename = utf8_path(path);
                return self.call([&](::CLAP *model) {
                    float *values = nullptr;
                    std::size_t count = 0;
                    check(clap_encode_audio_file(model, c_string(filename), crop_offset, &values, &count));
                    return take_buffer(values, count);
                });
            },
            py::arg("path"), py::arg("crop_offset") = 0)
        .def(
            "audio_frames",
            [](PythonCLAP &self, const std::vector<float> &pcm, int sample_rate) {
                validate_rate(sample_rate);
                validate_samples(pcm);
                return self.call([&](::CLAP *model) {
                    clap_audio_frames frames{};
                    check(clap_get_audio_frames(model, pcm.data(), pcm.size(), sample_rate, &frames));
                    const std::unique_ptr<clap_audio_frames, decltype(&clap_free_audio_frames)> owned(
                        &frames, clap_free_audio_frames);
                    std::vector<std::vector<float>> output;
                    output.reserve(frames.frames);
                    for (std::size_t i = 0; i < frames.frames; ++i) {
                        if (!frames.dimension)
                            output.emplace_back();
                        else
                            output.emplace_back(frames.values + i * frames.dimension,
                                                frames.values + (i + 1) * frames.dimension);
                    }
                    return output;
                });
            },
            py::arg("pcm"), py::arg("sample_rate") = 48000)
        .def("close", &PythonCLAP::close)
        .def(
            "__enter__",
            [](PythonCLAP &self) -> PythonCLAP & {
                self.call([](::CLAP *) {});
                return self;
            },
            py::return_value_policy::reference_internal)
        .def("__exit__", [](PythonCLAP &self, py::object, py::object, py::object) { self.close(); });
}
