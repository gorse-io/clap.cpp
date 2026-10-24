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

void embeddings(const httplib::Request &request, httplib::Response &response,
                clap::Model &model, const std::string &identifier, int dimensions,
                std::mutex &inference_mutex) {
    const auto body = json::parse(request.body, nullptr, false);
    if (!body.is_object() || !body.contains("input")) {
        invalid_request(response, "Expected a JSON object with an input field");
        return;
    }

    auto inputs = body["input"];
    if (inputs.is_string()) inputs = json::array({inputs});
    if (!inputs.is_array() || inputs.empty() || inputs.size() > 1024) {
        invalid_request(response, "input must be a non-empty string or an array of 1 to 1024 strings");
        return;
    }
    for (const auto &input : inputs) {
        if (!input.is_string() || input.get_ref<const std::string &>().empty()) {
            invalid_request(response, "Each input must be a non-empty string");
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
                const auto &text = inputs[index].get_ref<const std::string &>();
                const auto embedding = model.text(text);
                if (embedding.size() != static_cast<std::size_t>(dimensions))
                    throw std::runtime_error("Model returned an unexpected embedding dimension");
                for (float value : embedding)
                    if (!std::isfinite(value)) throw std::runtime_error("Model returned a non-finite embedding");
                data.push_back({{"object", "embedding"}, {"index", index}, {"embedding", embedding}});
                // Model::text(string) chooses its architecture's full context.
                // The public tokenize() default is 512 and can truncate or be
                // unsupported for smaller contexts. Omit usage rather than
                // report an inaccurate count in either situation.
                try {
                    const auto ids = model.tokenize(text);
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
        clap::Model model(model_path, threads);
        const auto identifier = std::filesystem::path(model_path).filename().string();
        const int dimensions = model.projection_dimension();
        if (dimensions <= 0) throw std::runtime_error("Model projection dimension must be positive");
        std::mutex inference_mutex;
        httplib::Server server;
        server.set_payload_max_length(1024 * 1024);
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
