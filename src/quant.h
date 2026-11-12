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
#pragma once

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>

struct ggml_tensor;

namespace clap::detail {
bool encoder_architecture(const std::string &architecture);
bool encoder_quantizable(const ggml_tensor *w, const std::string &architecture);

// Own an exclusively created file beside the destination. Publication is one
// no-replace hard-link operation, never an exists/rename pair. Filesystems that
// cannot create hard links fail safely instead of falling back to clobbering.
class QuantizeOutput {
  public:
    explicit QuantizeOutput(const std::filesystem::path &destination);
    QuantizeOutput(const QuantizeOutput &) = delete;
    QuantizeOutput &operator=(const QuantizeOutput &) = delete;
    ~QuantizeOutput();
    void write(const void *data, size_t size);
    void close();
    void publish();

  private:
    std::filesystem::path destination_, temporary_;
    std::FILE *stream_ = nullptr;
};
} // namespace clap::detail

int cmd_quantize(const std::string &input, const std::string &output, const std::string &type = "Q8_0");
