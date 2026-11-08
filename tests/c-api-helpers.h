#pragma once
#include "clap.h"

#include <memory>
#include <stdexcept>
#include <vector>

// Ownership/status helpers only: tests call each public C entry point directly.
namespace c_api_test {
using Model = std::unique_ptr<CLAP, decltype(&clap_destroy)>;
inline void check(clap_status status) {
    if (status != CLAP_OK)
        throw std::runtime_error(clap_last_error());
}
template <class T, class F> std::vector<T> take_buffer(F call) {
    T *data = nullptr;
    size_t count = 0;
    const auto status = call(&data, &count);
    std::unique_ptr<T, decltype(&clap_free)> owner(data, clap_free);
    check(status);
    if (!count)
        return {};
    return {data, data + count};
}
struct Frames {
    std::vector<float> values;
    size_t frames;
    size_t dimension;
};
template <class F> Frames take_frames(F call) {
    clap_audio_frames output{};
    const auto status = call(&output);
    std::unique_ptr<clap_audio_frames, decltype(&clap_free_audio_frames)> owner(&output, clap_free_audio_frames);
    check(status);
    Frames result{{}, output.frames, output.dimension};
    if (output.frames && output.dimension)
        result.values.assign(output.values, output.values + output.frames * output.dimension);
    return result;
}
} // namespace c_api_test
