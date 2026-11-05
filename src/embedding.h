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
#include <string>

struct EmbeddingOptions {
    std::string model_path;
    std::string mmproj_path;
    int threads = 0;
    std::string device;
    std::string text;
    std::string audio_path;
    std::string separator = "\n";
    // Empty means native single-vector JSON for --text/--audio, raw for --prompt/--file.
    std::string format;
    bool single_text = false;
    std::size_t crop_offset = 0;
    bool crop_requested = false;
    bool frames = false;
};

int cmd_embedding(const EmbeddingOptions &options);
