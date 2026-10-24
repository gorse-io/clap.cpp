#include "embedding.h"
#include "quantize.h"
#include "server.h"

#include <CLI/CLI.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    CLI::App app{"clap.cpp"};
    argv = app.ensure_utf8(argv);
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string server_model;
    int server_threads = 0;
    auto *serve = app.add_subcommand("serve", "Start the embeddings HTTP server");
    serve->add_option("-m,--model", server_model, "Converted model GGUF")
        ->required()->check(CLI::ExistingFile);
    serve->add_option("-t,--threads", server_threads, "ggml CPU threads (0: automatic)")
        ->check(CLI::NonNegativeNumber)->capture_default_str();
    serve->add_option("--host", host, "Listen address")->capture_default_str();
    serve->add_option("--port", port, "Listen port")
        ->check(CLI::Range(1, 65535))->capture_default_str();

    EmbeddingOptions options;
    std::string file;
    auto *embedding = app.add_subcommand("embedding", "Generate native text or audio embeddings");
    embedding->add_option("-m,--model", options.model_path, "Converted CLAP/CLaMP3/MERT/MERT2/MuQ/MuLan GGUF")
        ->required()->check(CLI::ExistingFile);
    embedding->add_option("-t,--threads", options.threads, "ggml CPU threads (0: automatic)")
        ->check(CLI::NonNegativeNumber)->capture_default_str();
    auto *input = embedding->add_option_group("Input", "Choose one text prompt, text file, or audio file");
    input->require_option(1, 1);
    input->add_option("-p,--prompt", options.text, "Separator-delimited text (default output: raw)");
    auto *text_option = input->add_option("--text", options.text, "One UTF-8 text (default output: flat JSON vector)");
    auto *file_option = input->add_option("-f,--file", file, "Read input text from a file")
        ->check(CLI::ExistingFile);
    auto *audio_option = input->add_option("-a,--audio", options.audio_path,
                                         "WAV, MP3 or FLAC (default output: flat JSON vector)")
        ->check(CLI::ExistingFile);
    embedding->add_option("--embd-separator", options.separator, "Prompt/file separator (default: newline)");
    embedding->add_option("--embd-output-format", options.format,
                          "Explicit output: raw rows, array of vectors, JSON list, or JSON list with cosine similarities")
        ->check(CLI::IsMember({"raw", "array", "json", "json+"}));
    auto *crop_option = embedding->add_option("--crop-offset", options.crop_offset,
                                            "Sample offset at 48 kHz into CLAP audio longer than ten seconds")
        ->check(CLI::NonNegativeNumber);
    embedding->add_flag("--frames", options.frames, "Return final-layer waveform-encoder frames instead of mean pooling");
    embedding->add_flag("--clamp3-features", options.clamp3_features,
                        "Return MERT-v1 mean-layer/time features (one vector per 5 s window)");

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

    try {
        if (*embedding) {
            options.single_text = text_option->count() != 0;
            options.crop_requested = crop_option->count() != 0;
            if (options.separator.empty()) throw std::runtime_error("Input separator must not be empty");
            if (options.crop_requested && !*audio_option)
                throw std::runtime_error("--crop-offset requires --audio");
            if (options.frames && (!*audio_option || options.crop_requested))
                throw std::runtime_error("--frames requires --audio and excludes --crop-offset");
            if (options.clamp3_features && (!*audio_option || options.frames || options.crop_requested))
                throw std::runtime_error("--clamp3-features requires --audio and excludes --frames/--crop-offset");
            if (*file_option) {
                std::ifstream stream(file, std::ios::binary);
                if (!stream) throw std::runtime_error("Failed to open input file: " + file);
                std::string contents;
                char buffer[4096];
                while (stream.read(buffer, sizeof(buffer))) {
                    contents.append(buffer, static_cast<std::size_t>(stream.gcount()));
                }
                if (stream.bad() || !stream.eof())
                    throw std::runtime_error("Failed to read input file: " + file);
                contents.append(buffer, static_cast<std::size_t>(stream.gcount()));
                options.text = contents;
            }
            if (!*audio_option && !options.single_text && options.text.empty())
                throw std::runtime_error("Input must not be empty");
            return run_embedding(options);
        }
        if (*serve) {
            return run_server(host, port, server_model, server_threads);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "clap.cpp\n";
    return 0;
}
