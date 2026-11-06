#include "server.h"
#include "server-scheduler.h"
#include "clap.h"
#include "audio.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

using nlohmann::json;

// Signal handlers cannot lock or touch HTTP/model objects. A normal thread
// performs shutdown; lock-free atomic stores are signal-safe.
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> stop_requested{false};
void request_stop(int) { stop_requested.store(true, std::memory_order_relaxed); }

class StopSignals {
  public:
    StopSignals() {
        stop_requested = false;
        interrupt_ = std::signal(SIGINT, request_stop);
        terminate_ = std::signal(SIGTERM, request_stop);
        if (interrupt_ == SIG_ERR || terminate_ == SIG_ERR) {
            restore();
            throw std::runtime_error("Failed to install server signal handlers");
        }
    }
    ~StopSignals() { restore(); }
    StopSignals(const StopSignals &) = delete;
    StopSignals &operator=(const StopSignals &) = delete;

  private:
    using Handler = void (*)(int);
    Handler interrupt_ = SIG_ERR;
    Handler terminate_ = SIG_ERR;
    void restore() {
        if (interrupt_ != SIG_ERR)
            std::signal(SIGINT, interrupt_);
        if (terminate_ != SIG_ERR)
            std::signal(SIGTERM, terminate_);
    }
};

class ShutdownWatch {
  public:
    explicit ShutdownWatch(std::function<void()> stop)
        : thread_([this, stop = std::move(stop)] {
              while (!done_.load(std::memory_order_relaxed)) {
                  if (stop_requested.load(std::memory_order_relaxed)) {
                      stop();
                      return;
                  }
                  std::this_thread::sleep_for(std::chrono::milliseconds(25));
              }
          }) {}
    ~ShutdownWatch() {
        done_.store(true, std::memory_order_relaxed);
        thread_.join();
    }
    ShutdownWatch(const ShutdownWatch &) = delete;
    ShutdownWatch &operator=(const ShutdownWatch &) = delete;

  private:
    std::atomic<bool> done_{false};
    std::thread thread_;
};

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

void error_response(httplib::Response &response, int status, const std::string &type, const std::string &message) {
    response.status = status;
    response.set_content(json{{"error", {{"message", message}, {"type", type}, {"code", status}}}}.dump(),
                         "application/json");
}

void invalid_request(httplib::Response &response, const std::string &message) {
    error_response(response, 400, "invalid_request_error", message);
}

// RFC 4648 standard alphabet, canonical padding and no ignored characters.
std::vector<unsigned char> decode_base64(const std::string &input) {
    if (input.empty() || input.size() % 4 != 0)
        throw std::invalid_argument("Audio data must be non-empty padded base64");
    const auto digit = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z')
            return c - 'A';
        if (c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if (c >= '0' && c <= '9')
            return c - '0' + 52;
        if (c == '+')
            return 62;
        if (c == '/')
            return 63;
        return -1;
    };
    std::vector<unsigned char> output;
    output.reserve(input.size() / 4 * 3);
    for (std::size_t i = 0; i < input.size(); i += 4) {
        const int a = digit(input[i]), b = digit(input[i + 1]);
        const int c = digit(input[i + 2]), d = digit(input[i + 3]);
        const bool pad2 = input[i + 2] == '=', pad3 = input[i + 3] == '=';
        if (a < 0 || b < 0 || (c < 0 && !pad2) || (d < 0 && !pad3) || (pad2 && !pad3) ||
            ((pad2 || pad3) && i + 4 != input.size()) || (pad2 && (b & 15) != 0) || (pad3 && !pad2 && (c & 3) != 0))
            throw std::invalid_argument("Invalid audio base64 alphabet or padding");
        output.push_back(static_cast<unsigned char>((a << 2) | (b >> 4)));
        if (!pad2)
            output.push_back(static_cast<unsigned char>((b << 4) | (c >> 2)));
        if (!pad3)
            output.push_back(static_cast<unsigned char>((c << 6) | d));
    }
    return output;
}

bool is_audio_input(const json &input) {
    if (!input.is_object() || !input.contains("content"))
        return false;
    const auto &content = input["content"];
    if (!content.is_array() || content.size() != 1)
        return false;
    const auto &block = content[0];
    if (!block.is_object() || block.value("type", json()) != "input_audio" || !block.contains("input_audio") ||
        !block["input_audio"].is_object())
        return false;
    const auto &audio = block["input_audio"];
    return audio.contains("data") && audio["data"].is_string();
}

using PreparedInput = std::variant<std::string, clap::Audio>;
struct InferenceResult {
    std::vector<std::vector<float>> embeddings;
    std::size_t tokens = 0;
    bool usage_available = true;
};
using InferenceQueue = clap::server::Scheduler<InferenceResult>;

// Only called on the dedicated owner thread. No HTTP request/response objects
// cross this boundary; the complete request remains one atomic queue item.
InferenceResult infer(CLAP *model, const std::vector<PreparedInput> &inputs, int dimensions,
                      const std::atomic<bool> &cancelled) {
    InferenceResult result;
    result.embeddings.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (cancelled)
            throw clap::server::Cancelled();
        const auto *text = std::get_if<std::string>(&inputs[index]);
        float *values = nullptr;
        std::size_t count = 0;
        if (text) {
            check(clap_encode_text(model, c_string(*text), &values, &count));
        } else {
            const auto &audio = std::get<clap::Audio>(inputs[index]);
            result.usage_available = false;
            check(clap_encode_audio(model, audio.samples.data(), audio.samples.size(), audio.sample_rate, 0, &values,
                                    &count));
        }
        auto embedding = take_buffer(values, count);
        if (cancelled)
            throw clap::server::Cancelled();
        if (embedding.size() != static_cast<std::size_t>(dimensions))
            throw std::runtime_error("Model returned an unexpected embedding dimension");
        for (float value : embedding)
            if (!std::isfinite(value))
                throw std::runtime_error("Model returned a non-finite embedding");
        result.embeddings.push_back(std::move(embedding));
        // The public tokenize() default may truncate or be unsupported for
        // smaller contexts. Omit usage rather than invent a token count.
        if (!text)
            continue;
        try {
            int *ids = nullptr;
            std::size_t size = 0;
            check(clap_tokenize(model, c_string(*text), 512, &ids, &size));
            const auto owned_ids = take_buffer(ids, size);
            if (owned_ids.size() >= 512)
                result.usage_available = false;
            else
                result.tokens += owned_ids.size();
        } catch (const std::exception &) {
            result.usage_available = false;
        }
    }
    return result;
}

// The pinned httplib callback polls then recv(MSG_PEEK)s, reporting both EOF
// and socket errors as "closed". EOF alone is ambiguous: a client may have
// shutdown(SHUT_WR) and still be waiting for its response. There is no public
// request socket/error API, so only cancel on a freshly reported fatal socket
// error from that callback. Do not send probe bytes or consume request data.
// Recheck this adapter when updating httplib's callback implementation.
bool connection_failed(const httplib::Request &request) {
#ifdef _WIN32
    const int previous = WSAGetLastError();
    WSASetLastError(0);
    const bool closed = request.is_connection_closed();
    const int error = WSAGetLastError();
    WSASetLastError(previous);
    return closed && (error == WSAECONNRESET || error == WSAECONNABORTED || error == WSAENETRESET ||
                      error == WSAENOTCONN || error == WSAENOTSOCK || error == WSAESHUTDOWN || error == WSAETIMEDOUT);
#else
    const int previous = errno;
    errno = 0;
    const bool closed = request.is_connection_closed();
    const int error = errno;
    errno = previous;
    return closed && (error == ECONNRESET || error == ECONNABORTED || error == ENETRESET || error == ENOTCONN ||
                      error == EBADF || error == ENOTSOCK || error == EPIPE || error == ETIMEDOUT);
#endif
}

void embeddings(const httplib::Request &request, httplib::Response &response, InferenceQueue &queue, CLAP *model,
                const std::string &identifier, int dimensions) {
    const auto body = json::parse(request.body, nullptr, false);
    if (!body.is_object() || !body.contains("input")) {
        invalid_request(response, "Expected a JSON object with an input field");
        return;
    }

    auto inputs = body["input"];
    if (inputs.is_string())
        inputs = json::array({inputs});
    if (!inputs.is_array() || inputs.empty() || inputs.size() > 1024) {
        invalid_request(response, "input must be a non-empty string or an array of 1 to 1024 text/audio inputs");
        return;
    }
    for (const auto &input : inputs) {
        if (!(input.is_string() && !input.get_ref<const std::string &>().empty()) && !is_audio_input(input)) {
            invalid_request(
                response,
                "Each input must be a non-empty string or exactly one input_audio content block with string data");
            return;
        }
    }
    for (const auto &input : inputs) {
        if (input.is_string() && input.get_ref<const std::string &>().find('\0') != std::string::npos) {
            invalid_request(response, "text must contain no NUL bytes");
            return;
        }
    }
    if (body.contains("model") && (!body["model"].is_string() || body["model"] != identifier)) {
        invalid_request(response, "model must match the loaded model: " + identifier);
        return;
    }
    if (body.contains("encoding_format") && body["encoding_format"] != "float") {
        invalid_request(response, "Only float encoding_format is supported");
        return;
    }
    if (body.contains("dimensions") && (!body["dimensions"].is_number_integer() || body["dimensions"] != dimensions)) {
        invalid_request(response, "dimensions must match the loaded model: " + std::to_string(dimensions));
        return;
    }

    try {
        // Validation and decoding are HTTP-side work, independent of the model.
        std::vector<PreparedInput> prepared;
        prepared.reserve(inputs.size());
        for (const auto &input : inputs) {
            if (connection_failed(request))
                return;
            if (input.is_string()) {
                prepared.emplace_back(input.get<std::string>());
            } else {
                try {
                    const auto bytes =
                        decode_base64(input["content"][0]["input_audio"]["data"].get_ref<const std::string &>());
                    prepared.emplace_back(clap::decode_audio(bytes.data(), bytes.size()));
                } catch (const std::exception &error) {
                    invalid_request(response, error.what());
                    return;
                }
            }
        }
        auto ticket = queue.submit([model, inputs = std::move(prepared), dimensions](const auto &cancelled) {
            return infer(model, inputs, dimensions, cancelled);
        });
        if (!ticket) {
            error_response(response, 503, "server_error", "Inference queue is full or stopping; retry later");
            return;
        }
        // Ticket destruction cancels on every exit path, including failures in
        // waiting/response construction. Only the HTTP thread inspects its
        // live request/socket; worker tasks never retain either by reference.
        while (ticket.future().wait_for(std::chrono::milliseconds(25)) != std::future_status::ready)
            if (connection_failed(request))
                return;
        const auto output = ticket.future().get();
        auto data = json::array();
        for (std::size_t index = 0; index < output.embeddings.size(); ++index)
            data.push_back({{"object", "embedding"}, {"index", index}, {"embedding", output.embeddings[index]}});
        json result = {{"object", "list"}, {"data", std::move(data)}, {"model", identifier}};
        if (output.usage_available)
            result["usage"] = {{"prompt_tokens", output.tokens}, {"total_tokens", output.tokens}};
        response.set_content(result.dump(), "application/json");
    } catch (const clap::server::Cancelled &error) {
        error_response(response, 503, "server_error", error.what());
    } catch (const std::exception &error) {
        error_response(response, 500, "server_error", error.what());
    } catch (...) {
        error_response(response, 500, "server_error", "Unknown inference error");
    }
}

} // namespace

int cmd_server(const std::string &host, int port, const std::string &model_path, int threads, const std::string &device,
               std::size_t queue_capacity) {
    try {
        // Startup waits for the owner thread to load the model before opening
        // the socket. The same worker also queries metadata and destroys it.
        const auto identifier = std::filesystem::path(model_path).filename().string();
        StopSignals signals;
        std::unique_ptr<CLAP, decltype(&clap_destroy)> model(nullptr, clap_destroy);
        int dimensions = 0;
        InferenceQueue queue(
            queue_capacity,
            [&] {
                auto options = clap_default_options();
                options.threads = threads;
                options.device = c_string(device);
                CLAP *handle = nullptr;
                check(clap_create(c_string(model_path), &options, &handle));
                model.reset(handle);
                dimensions = clap_projection_dimension(model.get());
                if (dimensions <= 0)
                    throw std::runtime_error("Model projection dimension must be positive");
            },
            [&] { model.reset(); });
        CLAP *const inference_model = model.get();
        // Destroy HTTP handlers first, then stop/join inference, then release
        // the enclosing model variable. Bind/listen failures use the same path.
        httplib::Server server;
        server.set_payload_max_length(32 * 1024 * 1024);
        server.Post("/v1/embeddings", [&](const httplib::Request &request, httplib::Response &response) {
            embeddings(request, response, queue, inference_model, identifier, dimensions);
        });
        server.set_exception_handler(
            [](const httplib::Request &, httplib::Response &response, std::exception_ptr exception) {
                try {
                    if (exception)
                        std::rethrow_exception(exception);
                } catch (const std::exception &error) {
                    error_response(response, 500, "server_error", error.what());
                    return;
                } catch (...) {
                }
                error_response(response, 500, "server_error", "Unknown server error");
            });
        if (!server.bind_to_port(host, port)) {
            std::cerr << "Failed to bind to " << host << ':' << port << '\n';
            return 1;
        }
        std::cout << "Listening on " << host << ':' << port << " with model " << identifier << std::endl;
        ShutdownWatch shutdown([&] {
            server.stop();
            queue.stop();
        });
        const bool listened = server.listen_after_bind();
        return listened || stop_requested.load(std::memory_order_relaxed) ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "Server startup failed: " << error.what() << '\n';
        return 1;
    }
}
