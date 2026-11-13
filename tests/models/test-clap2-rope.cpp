// CLAP2 size-geometry regression. Uses only the tiny analytical GGUF's
// constructor metadata; rotary inputs have the exact published tower widths.
#include "models/model-policy.h"
#include "models/models.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace clap::detail {
struct FeatureTestAccess {
    static ggml_tensor *rotate(const Clap2Model &model, Graph &graph, ggml_tensor *input, int heads, int time,
                               int prefix) {
        return model.rotate(graph, input, heads, 10000.f, time, prefix);
    }
};
} // namespace clap::detail

using namespace clap::detail;

static void rotary(const Clap2Model &model, const GraphRuntime &runtime, const char *name, int width, int heads,
                   bool audio) {
    // A non-square frequency/time grid detects swapped axes and flattening.
    // Cross both axes and exercise every head, including XXS's 56-wide heads.
    const int time = 7, frequency = 8, prefix = audio ? 1 : 0;
    const int tokens = audio ? time * frequency + prefix : 19;
    const int head_dim = width / heads;
    HostMatrix input(tokens, width);
    for (size_t i = 0; i < input.data.size(); ++i)
        input.data[i] = std::sin(float(i) * 0.17f) + 0.3f * std::cos(float(i) * 0.11f);
    Graph graph(runtime);
    auto *output = FeatureTestAccess::rotate(model, graph, graph.input(input), heads, audio ? time : 0, prefix);
    const auto actual = graph.run(output);
    if (actual.size() != input.data.size())
        throw std::runtime_error("CLAP2 rotary shape mismatch");
    float maximum = 0;
    // Scalar expression of upstream timm build_rotary_pos_embed: dim/4
    // inverse-frequency bands per axis, repeated interleaved. ModernText
    // uses dim/2 bands and ordinary positions. No power-of-two assumption.
    for (int token = 0; token < tokens; ++token)
        for (int head = 0; head < heads; ++head)
            for (int channel = 0; channel < head_dim; channel += 2) {
                const int axis_dim = audio ? head_dim / 2 : head_dim;
                const int within_axis = channel % axis_dim;
                const int position = audio ? (channel < axis_dim ? std::max(0, token - prefix) / time
                                                                 : std::max(0, token - prefix) % time)
                                           : token;
                const double inverse_frequency = std::exp(-std::log(10000.0) * within_axis / axis_dim);
                const double angle = token < prefix ? 0.0 : position * inverse_frequency;
                const int column = head * head_dim + channel;
                const double even = input[token][column], odd = input[token][column + 1];
                const double expected[] = {even * std::cos(angle) - odd * std::sin(angle),
                                           odd * std::cos(angle) + even * std::sin(angle)};
                for (int component = 0; component < 2; ++component) {
                    const float value = actual[size_t(token) * width + column + component];
                    if (!std::isfinite(value))
                        throw std::runtime_error("CLAP2 rotary produced non-finite output");
                    maximum = std::max(maximum, float(std::abs(value - expected[component])));
                    if (token < prefix && value != input[token][column + component])
                        throw std::runtime_error("CLAP2 audio register token was rotated");
                }
            }
    std::cout << name << (audio ? " audio" : " text") << " width=" << width << " heads=" << heads
              << " head_dim=" << head_dim << " max_abs_error=" << maximum << '\n';
    if (maximum > 3e-6f)
        throw std::runtime_error("CLAP2 rotary differs from independent scalar oracle");
}

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("usage: test-clap2-rope TINY-CLAP2.gguf");
        clap::RuntimeOptions options;
        options.device = "cpu";
        ModelLoader loader(options);
        clap::detail::load_model_weights(loader, argv[1]);
        Clap2Model model(loader, 2);
        GraphRuntime runtime(loader, 2, true);
        // Published config.json dimensions from pinned laion/clap2-*-audioxl-s1B.
        struct Size {
            const char *name;
            int audio_width, audio_heads, text_width, text_heads;
        };
        const Size sizes[] = {{"large", 1024, 16, 960, 15},
                              {"medium", 640, 10, 640, 10},
                              {"small", 512, 8, 512, 8},
                              {"xs", 320, 5, 320, 5},
                              {"xxs", 224, 4, 224, 4}};
        for (const auto &size : sizes) {
            rotary(model, runtime, size.name, size.audio_width, size.audio_heads, true);
            rotary(model, runtime, size.name, size.text_width, size.text_heads, false);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
