#include "server.h"

#include <CLI/CLI.hpp>

#include <iostream>
#include <string>

int main(int argc, char **argv) {
    CLI::App app{"clap.cpp"};
    std::string host = "127.0.0.1";
    int port = 8080;
    auto *serve = app.add_subcommand("serve", "Start the embeddings HTTP server");
    serve->add_option("--host", host, "Listen address")->capture_default_str();
    serve->add_option("--port", port, "Listen port")
        ->check(CLI::Range(1, 65535))->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    if (*serve) {
        return run_server(host, port);
    }
    std::cout << "clap.cpp\n";
    return 0;
}
