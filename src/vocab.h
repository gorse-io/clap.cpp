#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>
#include <unicode/regex.h>

namespace clap::detail {

// Shared RoBERTa/GPT-2 byte-level BPE implementation, independent of weight I/O
// and encoder execution. ICU owns Unicode pre-tokenization only.
class ByteBpeVocabulary {
public:
    ByteBpeVocabulary(const nlohmann::json &tokenizer, int bos, int eos, size_t context_length);
    std::vector<int> tokenize(const std::string &text, size_t max_length) const;
private:
    void encode_piece(const std::string &text, std::vector<int> &ids) const;
    int bos, eos;
    size_t context_length;
    std::unordered_map<std::string, int> vocab;
    std::map<std::pair<std::string, std::string>, int> ranks;
    std::vector<nlohmann::json> special;
    std::string byte_chars[256];
    std::unique_ptr<icu::RegexPattern> pattern;
};
} // namespace clap::detail
