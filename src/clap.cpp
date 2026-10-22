#include "embedding.h"
#include "quantize.h"
#include "server.h"

#include <CLI/CLI.hpp>

#include <fstream>
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

    std::string prompt;
    std::string file;
    std::string separator = "\n";
    std::string format = "raw";
    auto *embedding = app.add_subcommand("embedding", "Generate dummy text embeddings");
    auto *input = embedding->add_option_group("Input", "Choose a prompt or a UTF-8 text file");
    input->require_option(1, 1);
    input->add_option("-p,--prompt", prompt, "Input text");
    auto *file_option = input->add_option("-f,--file", file, "Read input text from a file")
        ->check(CLI::ExistingFile);
    embedding->add_option("--embd-separator", separator, "Input separator (default: newline)");
    embedding->add_option("--embd-output-format", format, "Output format")
        ->check(CLI::IsMember({"raw", "array", "json", "json+"}))->capture_default_str();

    std::string quantize_input;
    std::string quantize_output;
    std::string quantize_type = "Q8_0";
    auto *quantize = app.add_subcommand("quantize", "Convert F32/F16 GGUF to Q8_0 or mixed Q4_K_M");
    quantize->add_option("--type", quantize_type, "Quantization preset")
        ->check(CLI::IsMember({"Q8_0", "Q4_K_M"}, CLI::ignore_case))
        ->capture_default_str();
    quantize->add_option("INPUT.gguf", quantize_input, "F32/F16 CLAP/CLaMP3/MERT/MERT2/MuQ/MuLan model")->required();
    quantize->add_option("OUTPUT.gguf", quantize_output, "Output model (must not exist)")->required();

    CLI11_PARSE(app, argc, argv);

    if (*quantize) {
        return run_quantize(quantize_input, quantize_output, quantize_type);
    }

    if (*embedding) {
        if (separator.empty()) {
            std::cerr << "Input separator must not be empty\n";
            return 1;
        }
        if (*file_option) {
            std::ifstream stream(file, std::ios::binary);
            if (!stream) {
                std::cerr << "Failed to open input file: " << file << '\n';
                return 1;
            }
            std::string contents;
            char buffer[4096];
            while (stream.read(buffer, sizeof(buffer))) {
                contents.append(buffer, static_cast<std::size_t>(stream.gcount()));
            }
            if (stream.bad() || !stream.eof()) {
                std::cerr << "Failed to read input file: " << file << '\n';
                return 1;
            }
            contents.append(buffer, static_cast<std::size_t>(stream.gcount()));
            prompt = contents;
        }
        if (prompt.empty()) {
            std::cerr << "Input must not be empty\n";
            return 1;
        }
        return run_embedding(prompt, separator, format);
    }

    if (*serve) {
        return run_server(host, port);
    }
    std::cout << "clap.cpp\n";
    return 0;
}
