#include "server.h"
#include "clap.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

using nlohmann::json;

void error_response(httplib::Response &response, int status,
                    const std::string &type, const std::string &message) {
    response.status = status;
    response.set_content(json{{"error", {
        {"message", message}, {"type", type}, {"code", status}
    }}}.dump(), "application/json");
}

void invalid_request(httplib::Response &response, const std::string &message) {
    error_response(response, 400, "invalid_request_error", message);
}

// RFC 4648 standard alphabet, canonical padding and no ignored characters.
std::vector<unsigned char> decode_base64(const std::string &input) {
    if (input.empty() || input.size() % 4 != 0)
        throw std::invalid_argument("Audio data must be non-empty padded base64");
    const auto digit = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<unsigned char> output;
    output.reserve(input.size() / 4 * 3);
    for (std::size_t i = 0; i < input.size(); i += 4) {
        const int a = digit(input[i]), b = digit(input[i + 1]);
        const int c = digit(input[i + 2]), d = digit(input[i + 3]);
        const bool pad2 = input[i + 2] == '=', pad3 = input[i + 3] == '=';
        if (a < 0 || b < 0 || (c < 0 && !pad2) || (d < 0 && !pad3) ||
            (pad2 && !pad3) || ((pad2 || pad3) && i + 4 != input.size()) ||
            (pad2 && (b & 15) != 0) || (pad3 && !pad2 && (c & 3) != 0))
            throw std::invalid_argument("Invalid audio base64 alphabet or padding");
        output.push_back(static_cast<unsigned char>((a << 2) | (b >> 4)));
        if (!pad2) output.push_back(static_cast<unsigned char>((b << 4) | (c >> 2)));
        if (!pad3) output.push_back(static_cast<unsigned char>((c << 6) | d));
    }
    return output;
}

bool is_audio_input(const json &input) {
    if (!input.is_object() || !input.contains("content")) return false;
    const auto &content = input["content"];
    if (!content.is_array() || content.size() != 1) return false;
    const auto &block = content[0];
    if (!block.is_object() || block.value("type", json()) != "input_audio" ||
        !block.contains("input_audio") || !block["input_audio"].is_object()) return false;
    const auto &audio = block["input_audio"];
    return audio.contains("data") && audio["data"].is_string();
}

void embeddings(const httplib::Request &request, httplib::Response &response,
                clap::CLAP &model, const std::string &identifier, int dimensions,
                std::mutex &inference_mutex) {
    const auto body = json::parse(request.body, nullptr, false);
    if (!body.is_object() || !body.contains("input")) {
        invalid_request(response, "Expected a JSON object with an input field");
        return;
    }

    auto inputs = body["input"];
    if (inputs.is_string()) inputs = json::array({inputs});
    if (!inputs.is_array() || inputs.empty() || inputs.size() > 1024) {
        invalid_request(response, "input must be a non-empty string or an array of 1 to 1024 text/audio inputs");
        return;
    }
    for (const auto &input : inputs) {
        if (!(input.is_string() && !input.get_ref<const std::string &>().empty()) && !is_audio_input(input)) {
            invalid_request(response, "Each input must be a non-empty string or exactly one input_audio content block with string data");
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
    if (body.contains("dimensions") &&
        (!body["dimensions"].is_number_integer() || body["dimensions"] != dimensions)) {
        invalid_request(response, "dimensions must match the loaded model: " + std::to_string(dimensions));
        return;
    }

    try {
        auto data = json::array();
        std::size_t tokens = 0;
        bool usage_available = true;
        {
            // Vocabulary and inference both belong to the same non-thread-safe
            // handle. Keep the complete batch atomic with respect to other calls.
            const std::lock_guard<std::mutex> lock(inference_mutex);
            for (std::size_t index = 0; index < inputs.size(); ++index) {
                const auto &input = inputs[index];
                std::vector<float> embedding;
                if (input.is_string()) {
                    embedding = model.encode_text(input.get_ref<const std::string &>());
                } else {
                    clap::Audio audio;
                    // Keep decoder errors separate from model inference errors.
                    try {
                        const auto bytes = decode_base64(input["content"][0]["input_audio"]["data"].get_ref<const std::string &>());
                        audio = clap::decode_audio(bytes.data(), bytes.size());
                    } catch (const std::exception &error) {
                        invalid_request(response, error.what());
                        return;
                    }
                    usage_available = false;
                    embedding = model.encode_audio(audio.samples, audio.sample_rate);
                }
                if (embedding.size() != static_cast<std::size_t>(dimensions))
                    throw std::runtime_error("Model returned an unexpected embedding dimension");
                for (float value : embedding)
                    if (!std::isfinite(value)) throw std::runtime_error("Model returned a non-finite embedding");
                data.push_back({{"object", "embedding"}, {"index", index}, {"embedding", embedding}});
                // CLAP::encode_text(string) chooses its architecture's full context.
                // The public tokenize() default is 512 and can truncate or be
                // unsupported for smaller contexts. Omit usage rather than
                // report an inaccurate count in either situation.
                if (!input.is_string()) continue;
                try {
                    const auto ids = model.tokenize(input.get_ref<const std::string &>());
                    if (ids.size() >= 512) usage_available = false;
                    else tokens += ids.size();
                } catch (const std::exception &) {
                    usage_available = false;
                }
            }
        }
        json result = {{"object", "list"}, {"data", data}, {"model", identifier}};
        if (usage_available)
            result["usage"] = {{"prompt_tokens", tokens}, {"total_tokens", tokens}};
        response.set_content(result.dump(), "application/json");
    } catch (const std::exception &error) {
        error_response(response, 500, "server_error", error.what());
    } catch (...) {
        error_response(response, 500, "server_error", "Unknown inference error");
    }
}

} // namespace

int run_server(const std::string &host, int port, const std::string &model_path, int threads) {
    try {
        // Load once, before opening the listening socket. A bad model must fail
        // startup rather than leave a live server that can never serve requests.
        clap::CLAP model(model_path, threads);
        const auto identifier = std::filesystem::path(model_path).filename().string();
        const int dimensions = model.projection_dimension();
        if (dimensions <= 0) throw std::runtime_error("Model projection dimension must be positive");
        std::mutex inference_mutex;
        httplib::Server server;
        server.set_payload_max_length(32 * 1024 * 1024);
        server.Post("/v1/embeddings", [&](const httplib::Request &request, httplib::Response &response) {
            embeddings(request, response, model, identifier, dimensions, inference_mutex);
        });
        server.set_exception_handler([](const httplib::Request &, httplib::Response &response,
                                        std::exception_ptr exception) {
            try {
                if (exception) std::rethrow_exception(exception);
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
        return server.listen_after_bind() ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "Server startup failed: " << error.what() << '\n';
        return 1;
    }
}
