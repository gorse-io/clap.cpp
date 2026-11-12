/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "clap.h"
#include "embedding.h"
#include "quantize.h"
#include "search.h"
#include "server.h"

#include <CLI/CLI.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    CLI::App app{"clap.cpp"};
    argv = app.ensure_utf8(argv);
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string server_model;
    int server_threads = 0;
    std::string server_device;
    auto *devices = app.add_subcommand("devices", "List available GGML devices");
    auto *serve = app.add_subcommand("serve", "Start the embeddings HTTP server");
    serve->add_option("-m,--model", server_model, "Converted model GGUF")->required()->check(CLI::ExistingFile);
    serve->add_option("-t,--threads", server_threads, "ggml CPU threads (0: hardware concurrency)")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();
    serve->add_option("--device", server_device, "GGML device name, auto or cpu (default: CLAP_DEVICE or auto)");
    serve->add_option("--host", host, "Listen address")->capture_default_str();
    serve->add_option("--port", port, "Listen port")->check(CLI::Range(1, 65535))->capture_default_str();

    EmbeddingOptions options;
    auto *embedding = app.add_subcommand("embedding", "Generate native text or audio embeddings");
    embedding
        ->add_option("-m,--model", options.model_path, "CLAP/CLaMP3/MERT/MERT2/MuQ/MuLan GGUF or standard Qwen3 GGUF")
        ->required()
        ->check(CLI::ExistingFile);
    embedding->add_option("-t,--threads", options.threads, "ggml CPU threads (0: hardware concurrency)")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();
    embedding->add_option("--device", options.device, "GGML device name, auto or cpu (default: CLAP_DEVICE or auto)");
    auto *input = embedding->add_option_group("Input", "Choose one text or audio file");
    input->require_option(1, 1);
    input->add_option("--text", options.text, "One UTF-8 text (output: flat JSON vector)");
    auto *audio_option =
        input->add_option("-a,--audio", options.audio_path, "WAV, MP3 or FLAC (output: flat JSON vector)")
            ->check(CLI::ExistingFile);
    auto *crop_option = embedding
                            ->add_option("--crop-offset", options.crop_offset,
                                         "Sample offset at 48 kHz into CLAP audio longer than ten seconds")
                            ->check(CLI::NonNegativeNumber);
    embedding->add_flag("--frames", options.frames,
                        "Return final-layer waveform-encoder frames instead of mean pooling");

    SearchOptions search_options;
    auto *search = app.add_subcommand("search", "Search a directory recursively for similar audio");
    search->add_option("DIRECTORY", search_options.directory, "Directory of WAV, MP3 and FLAC files")->required();
    search->add_option("-m,--model", search_options.model_path, "Converted model GGUF")
        ->required()
        ->check(CLI::ExistingFile);
    search->add_option("-k,--top-k", search_options.top_k, "Maximum number of results")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    search->add_option("-t,--threads", search_options.threads, "ggml CPU threads (0: hardware concurrency)")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();
    search->add_option("--device", search_options.device,
                       "GGML device name, auto or cpu (default: CLAP_DEVICE or auto)");
    auto *search_input = search->add_option_group("Input", "Choose one text or audio query");
    search_input->require_option(1, 1);
    auto *search_text = search_input->add_option("--text", search_options.text, "UTF-8 text query");
    search_input->add_option("--audio", search_options.audio_path, "WAV, MP3 or FLAC query")->check(CLI::ExistingFile);

    std::string quantize_input;
    std::string quantize_output;
    std::string quantize_type = "Q8_0";
    auto *quantize = app.add_subcommand("quantize", "Convert F32/F16/BF16 GGUF to BF16, Q8_0 or mixed Q4_K_M");
    quantize->add_option("--type", quantize_type, "Quantization preset")
        ->check(CLI::IsMember({"BF16", "Q8_0", "Q4_K_M"}, CLI::ignore_case))
        ->capture_default_str();
    quantize->add_option("INPUT.gguf", quantize_input, "F32/F16/BF16 CLAP/CLaMP3/MERT/MERT2/MuQ/MuLan model")
        ->required();
    quantize->add_option("OUTPUT.gguf", quantize_output, "Output model (must not exist)")->required();

    CLI11_PARSE(app, argc, argv);

    if (*quantize) {
        return cmd_quantize(quantize_input, quantize_output, quantize_type);
    }

    try {
        if (*devices) {
            clap_device *list = nullptr;
            std::size_t count = 0;
            if (clap_available_devices(&list, &count) != CLAP_OK)
                throw std::runtime_error(clap_last_error());
            const auto free_devices = [count](clap_device *value) { clap_free_devices(value, count); };
            const std::unique_ptr<clap_device, decltype(free_devices)> owned(list, free_devices);
            for (std::size_t i = 0; i < count; ++i)
                std::cout << list[i].name << "\t" << list[i].description << "\n";
            return 0;
        }
        if (*embedding) {
            options.crop_requested = crop_option->count() != 0;
            if (options.crop_requested && !*audio_option)
                throw std::runtime_error("--crop-offset requires --audio");
            if (options.frames && (!*audio_option || options.crop_requested))
                throw std::runtime_error("--frames requires --audio and excludes --crop-offset");
            return cmd_embedding(options);
        }
        if (*search) {
            search_options.text_requested = search_text->count() != 0;
            return cmd_search(search_options);
        }
        if (*serve) {
            return cmd_server(host, port, server_model, server_threads, server_device);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << app.help();
    return 0;
}
