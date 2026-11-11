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

#include <nlohmann/json.hpp>
#include <unicode/regex.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace clap::detail {
class ModelLoader;

class Vocabulary {
  public:
    virtual ~Vocabulary() = default;
    virtual std::vector<int> tokenize(const std::string &text, size_t max_length) const = 0;
};

// Shared RoBERTa/GPT-2 byte-level BPE implementation, independent of weight I/O
// and encoder execution. ICU owns Unicode pre-tokenization only.
class ByteBpeVocabulary final : public Vocabulary {
  public:
    ByteBpeVocabulary(const nlohmann::json &tokenizer, int bos, int eos, size_t context_length,
                      bool split_special_tokens = false, bool qwen_rules = false);
    ByteBpeVocabulary(const ModelLoader &loader, size_t context_length);
    std::vector<int> tokenize(const std::string &text, size_t max_length) const override;

  private:
    void encode_piece(const std::string &text, std::vector<int> &ids) const;
    int bos, eos;
    bool qwen; // Standard GGUF qwen2 pre-tokenizer; no HF NFC normalization.
    size_t context_length;
    std::unordered_map<std::string, int> vocab;
    std::map<std::pair<std::string, std::string>, int> ranks;
    std::vector<nlohmann::json> special;
    std::string byte_chars[256];
    std::unique_ptr<icu::RegexPattern> pattern;
};

// XLM-R's tokenizer.json is a SentencePiece Unigram model represented by the
// Hugging Face fast-tokenizer format.  Inference stays self-contained: GGUF
// carries that JSON and ICU supplies the Unicode normalization/whitespace pass.
class UnigramVocabulary final : public Vocabulary {
  public:
    UnigramVocabulary(const nlohmann::json &tokenizer, int bos, int eos, int unk, size_t context_length);
    std::vector<int> tokenize(const std::string &text, size_t max_length) const override;

  private:
    struct Piece {
        int id;
        double score;
    };
    void encode_word(const std::string &text, std::vector<int> &ids) const;
    std::string normalize(const std::string &text) const;
    bool transform(const std::string &text, std::string &result) const;
    int bos, eos, unk;
    size_t context_length;
    size_t longest_piece = 0;
    double unknown_score = 0;
    std::vector<std::uint32_t> normalization_trie;
    std::string normalization_replacements;
    std::unordered_map<std::string, Piece> pieces;
    std::unordered_map<std::string, int> special;
};
} // namespace clap::detail
