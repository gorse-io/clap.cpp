#include "server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <iostream>

namespace {

using nlohmann::json;

void invalid_request(httplib::Response &response, const std::string &message) {
    response.status = 400;
    response.set_content(json{{"error", {
        {"message", message}, {"type", "invalid_request_error"}, {"code", 400}
    }}}.dump(), "application/json");
}

void embeddings(const httplib::Request &request, httplib::Response &response) {
    const auto body = json::parse(request.body, nullptr, false);
    if (!body.is_object() || !body.contains("input")) {
        invalid_request(response, "Expected a JSON object with an input field");
        return;
    }

    auto inputs = body["input"];
    if (inputs.is_string()) {
        inputs = json::array({inputs});
    }
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
    if (body.contains("model") && !body["model"].is_string()) {
        invalid_request(response, "model must be a string");
        return;
    }
    if (body.contains("encoding_format") && body["encoding_format"] != "float") {
        invalid_request(response, "Only float encoding_format is supported");
        return;
    }
    if (body.contains("dimensions") &&
        (!body["dimensions"].is_number_integer() || body["dimensions"] != 3)) {
        invalid_request(response, "Only 3 dimensions are supported");
        return;
    }

    auto data = json::array();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        data.push_back({
            {"object", "embedding"}, {"index", index},
            {"embedding", {1.0f, 0.0f, 0.0f}}
        });
    }
    response.set_content(json{
        {"object", "list"}, {"data", data}, {"model", body.value("model", "dummy")},
        {"usage", {{"prompt_tokens", 0}, {"total_tokens", 0}}}
    }.dump(), "application/json");
}

} // namespace

int run_server(const std::string &host, int port) {
    httplib::Server server;
    server.set_payload_max_length(1024 * 1024);
    server.Post("/v1/embeddings", embeddings);
    if (!server.bind_to_port(host, port)) {
        std::cerr << "Failed to bind to " << host << ':' << port << '\n';
        return 1;
    }
    std::cout << "Listening on " << host << ':' << port << std::endl;
    return server.listen_after_bind() ? 0 : 1;
}
