#include "vocab.h"
#include "impl.h"
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <cctype>
#include <limits>

namespace clap::detail {
using json = nlohmann::json;
namespace {
std::string utf8(int cp) {
    icu::UnicodeString u; u.append(static_cast<UChar32>(cp));
    std::string s; u.toUTF8String(s); return s;
}
} // namespace

ByteBpeVocabulary::ByteBpeVocabulary(const json &tokenizer, int bos, int eos, size_t context_length)
    : bos(bos), eos(eos), context_length(context_length) {
    check(tokenizer.at("model").at("type") == "BPE", "expected byte-level BPE");
    for (auto it = tokenizer["model"]["vocab"].begin(); it != tokenizer["model"]["vocab"].end(); ++it)
        vocab.emplace(it.key(), it.value().get<int>());
    int rank = 0;
    for (const auto &entry : tokenizer["model"]["merges"]) {
        std::string x, y;
        if (entry.is_array()) { x = entry.at(0); y = entry.at(1); }
        else {
            const std::string s = entry; const auto space = s.find(' ');
            check(space != std::string::npos, "invalid BPE merge"); x = s.substr(0, space); y = s.substr(space + 1);
        }
        ranks.emplace(std::make_pair(x, y), rank++);
    }
    if (tokenizer.contains("added_tokens")) for (const auto &v : tokenizer["added_tokens"]) {
        check(!v.at("content").get<std::string>().empty(), "empty added token");
        special.push_back(v);
    }
    int extra = 256;
    for (int i = 0; i < 256; ++i) {
        const bool regular = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174;
        byte_chars[i] = utf8(regular ? i : extra++);
    }
    UErrorCode status = U_ZERO_ERROR;
    pattern.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"), 0, status));
    check(U_SUCCESS(status), "cannot compile tokenizer Unicode regex");
}

void ByteBpeVocabulary::encode_piece(const std::string &text, std::vector<int> &ids) const {
    UErrorCode status = U_ZERO_ERROR;
    const auto unicode = icu::UnicodeString::fromUTF8(text);
    std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(unicode, status));
    check(U_SUCCESS(status), "tokenizer regex failed");
    while (matcher->find(status)) {
        std::string piece; matcher->group(status).toUTF8String(piece);
        std::vector<std::string> symbols;
        for (unsigned char b : piece) symbols.push_back(byte_chars[b]);
        while (symbols.size() > 1) {
            int best = std::numeric_limits<int>::max();
            std::pair<std::string, std::string> pair;
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                auto it = ranks.find({symbols[i], symbols[i + 1]});
                if (it != ranks.end() && it->second < best) { best = it->second; pair = it->first; }
            }
            if (best == std::numeric_limits<int>::max()) break;
            std::vector<std::string> merged;
            for (size_t i = 0; i < symbols.size(); ++i) {
                if (i + 1 < symbols.size() && symbols[i] == pair.first && symbols[i + 1] == pair.second) {
                    merged.push_back(symbols[i] + symbols[i + 1]);
                    ++i;
                } else merged.push_back(symbols[i]);
            }
            symbols = std::move(merged);
        }
        for (const auto &symbol : symbols) {
            const auto it = vocab.find(symbol);
            check(it != vocab.end(), "tokenizer has no token for byte sequence"); ids.push_back(it->second);
        }
    }
    check(U_SUCCESS(status), "tokenizer regex failed");
}

std::vector<int> ByteBpeVocabulary::tokenize(const std::string &text, size_t max_length) const {
    check(max_length >= 2 && max_length <= context_length, "invalid maximum text length");
    std::vector<int> ids{bos};
    size_t pos = 0;
    while (pos < text.size()) {
        size_t next = std::string::npos;
        const json *match = nullptr;
        for (const auto &s : special) {
            const auto p = text.find(s.at("content").get<std::string>(), pos);
            if (p < next) { next = p; match = &s; }
        }
        if (!match) { encode_piece(text.substr(pos), ids); break; }
        size_t end = next;
        // RoBERTa <mask> consumes left whitespace; Unicode handling is required.
        if (match->value("lstrip", false)) {
            auto u = icu::UnicodeString::fromUTF8(text.substr(pos, next - pos));
            while (!u.isEmpty()) {
                const UChar32 cp = u.char32At(u.moveIndex32(u.length(), -1));
                if (!u_isUWhiteSpace(cp)) break;
                u.truncate(u.moveIndex32(u.length(), -1));
            }
            std::string prefix; u.toUTF8String(prefix); end = pos + prefix.size();
        }
        encode_piece(text.substr(pos, end - pos), ids);
        ids.push_back(match->at("id").get<int>());
        pos = next + match->at("content").get<std::string>().size();
        if (match->value("rstrip", false)) {
            while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
        }
    }
    if (ids.size() > max_length - 1) ids.resize(max_length - 1);
    ids.push_back(eos);
    return ids;
}
} // namespace clap::detail
