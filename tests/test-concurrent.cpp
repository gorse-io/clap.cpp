#include "clap.h"
#include "context.h"

#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
struct Gate {
    std::mutex mutex;
    std::condition_variable condition;
    int entered = 0;
    bool release = false;
};
class Probe final : public clap::detail::ModelArchitecture {
    const clap::detail::ModelLoader &loader;
    std::unique_ptr<const clap::detail::ModelArchitecture> original;
    Gate &gate;
    std::vector<float> run(clap::detail::Context &context, const std::vector<int> &ids) const {
        return context.runtime(this, loader)
            .run_cached("per-call-isolation", {clap::detail::GraphInput(ids)},
                        [&](clap::detail::Graph &graph) {
                            {
                                std::unique_lock<std::mutex> lock(gate.mutex);
                                ++gate.entered;
                                gate.condition.notify_all();
                                gate.condition.wait(lock, [&] { return gate.release; });
                            }
                            return std::vector<ggml_tensor *>{graph.embedding(
                                loader.weight("text.embeddings.word_embeddings.weight"), graph.dynamic_input(0))};
                        })
            .at(0)
            .values;
    }

  public:
    Probe(const clap::detail::ModelLoader &source, std::unique_ptr<const clap::detail::ModelArchitecture> model,
          Gate &rendezvous)
        : loader(source), original(std::move(model)), gate(rendezvous) {}
    void initialize_context(clap::detail::Context &context) const override { context.runtime(this, loader); }
    std::vector<float> text(clap::detail::Context &context, const std::vector<int> &ids) const override {
        return run(context, ids);
    }
    clap::AudioFrames audio_frames(clap::detail::Context &, const std::vector<float> &) const override {
        throw std::runtime_error("unused");
    }
    std::vector<float> waveform_embedding(clap::detail::Context &context, const std::vector<float> &,
                                          size_t) const override {
        return run(context, {1, 2, 3});
    }
    clap::detail::TextParameters text_parameters() const override { return original->text_parameters(); }
    int projection_dimension() const override { return original->projection_dimension(); }
    int mel_bins() const override { return original->mel_bins(); }
    int sample_rate() const override { return original->sample_rate(); }
    bool is_waveform_encoder() const override { return false; }
    bool has_waveform_embeddings() const override { return true; }

  private:
    void load_arch_hparams() override {}
    void load_arch_tensors() override {}
};
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: test-concurrent model.gguf");
        clap::RuntimeOptions options{1, "cpu"};
        auto shared = std::make_shared<clap::detail::Model>(argv[1], options);
        Gate gate;
        shared->architecture = std::make_unique<Probe>(shared->loader, std::move(shared->architecture), gate);
        CLAP handle(shared);
        auto text = std::async(std::launch::async, [&] {
            float *data = nullptr;
            size_t count = 0;
            const auto status = clap_encode_text(&handle, "hello", &data, &count);
            std::unique_ptr<float, decltype(&clap_free)> owned(data, clap_free);
            require(status == CLAP_OK && data && count, "parallel text failed");
            return std::vector<float>(data, data + count);
        });
        auto audio = std::async(std::launch::async, [&] {
            const float samples[]{0.1f, 0.2f, 0.3f};
            float *data = nullptr;
            size_t count = 0;
            const auto status = clap_encode_audio(&handle, samples, 3, 48000, 0, &data, &count);
            std::unique_ptr<float, decltype(&clap_free)> owned(data, clap_free);
            require(status == CLAP_OK && data && count, "parallel audio failed");
            return std::vector<float>(data, data + count);
        });
        bool overlap;
        {
            std::unique_lock<std::mutex> lock(gate.mutex);
            overlap = gate.condition.wait_for(lock, std::chrono::seconds(3), [&] { return gate.entered == 2; });
            gate.release = true;
        }
        gate.condition.notify_all();
        const auto a = text.get(), b = audio.get();
        require(overlap, "same-handle text/audio share an execution lock instead of private per-call contexts");
        require(shared->weights->stats().host_tensors == 1, "concurrent calls duplicated shared model weights");
        auto *table = shared->loader.weight("text.embeddings.word_embeddings.weight");
        const auto *values = static_cast<const float *>(table->data);
        require(b.size() == size_t(3 * table->ne[0]), "audio result shape wrong");
        for (size_t i = 0; i < b.size(); ++i)
            require(b[i] == values[table->ne[0] + i], "parallel output corrupted");
        std::cout << "same-handle text/audio builders overlap with one immutable weight buffer\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
