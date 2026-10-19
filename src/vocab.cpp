#include "vocab.h"
#include "impl.h"
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>

namespace clap::detail {
using json = nlohmann::json;
namespace {
std::string utf8(int cp) {
    icu::UnicodeString u; u.append(static_cast<UChar32>(cp));
    std::string s; u.toUTF8String(s); return s;
}

std::vector<unsigned char> decode_charsmap(const std::string &encoded) {
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    check(!encoded.empty() && encoded.size() % 4 == 0, "invalid tokenizer charsmap encoding");
    std::vector<unsigned char> bytes;
    for (size_t i = 0; i < encoded.size(); i += 4) {
        uint32_t value = 0;
        int padding = 0;
        for (size_t j = 0; j < 4; ++j) {
            const char c = encoded[i + j];
            if (c == '=') {
                check(i + 4 == encoded.size() && j >= 2, "invalid tokenizer charsmap padding");
                ++padding; value <<= 6;
            } else {
                const auto digit = alphabet.find(c);
                check(!padding && digit != std::string::npos, "invalid tokenizer charsmap encoding");
                value = (value << 6) | static_cast<uint32_t>(digit);
            }
        }
        bytes.push_back(static_cast<unsigned char>(value >> 16));
        if (padding < 2) bytes.push_back(static_cast<unsigned char>(value >> 8));
        if (!padding) bytes.push_back(static_cast<unsigned char>(value));
    }
    return bytes;
}

uint32_t read_le32(const unsigned char *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint32_t trie_offset(uint32_t unit) {
    return (unit >> 10) << ((unit & (1u << 9)) >> 6);
}

// Boundaries needed by HF's short-grapheme character-map pass. ICU's bundled
// Unicode-property tables are sufficient; ICU BreakIterator and regex \\X need
// locale data excluded from our static build. InCB conjunct sequences only
// contain >=3-byte scalars, so splitting them cannot change the <6-byte pass:
// both paths normalize their individual scalars.
std::vector<int32_t> normalizer_boundaries(const icu::UnicodeString &text) {
    struct Character { int32_t start; UChar32 value; int property; };
    std::vector<Character> chars;
    for (int32_t i = 0; i < text.length();) {
        const auto value = text.char32At(i);
        chars.push_back({i, value, u_getIntPropertyValue(value, UCHAR_GRAPHEME_CLUSTER_BREAK)});
        i = text.moveIndex32(i, 1);
    }
    std::vector<int32_t> boundaries{0};
    auto control = [](int p) { return p == U_GCB_CONTROL || p == U_GCB_CR || p == U_GCB_LF; };
    size_t regional = !chars.empty() && chars[0].property == U_GCB_REGIONAL_INDICATOR;
    for (size_t i = 1; i < chars.size(); ++i) {
        const int a = chars[i - 1].property, b = chars[i].property;
        bool join = false;
        if (a == U_GCB_CR && b == U_GCB_LF) join = true;
        else if (control(a) || control(b)) join = false;
        else if (a == U_GCB_L && (b == U_GCB_L || b == U_GCB_V || b == U_GCB_LV || b == U_GCB_LVT)) join = true;
        else if ((a == U_GCB_LV || a == U_GCB_V) && (b == U_GCB_V || b == U_GCB_T)) join = true;
        else if ((a == U_GCB_LVT || a == U_GCB_T) && b == U_GCB_T) join = true;
        else if (b == U_GCB_EXTEND || b == U_GCB_ZWJ || b == U_GCB_SPACING_MARK || a == U_GCB_PREPEND) join = true;
        else if (a == U_GCB_REGIONAL_INDICATOR && b == U_GCB_REGIONAL_INDICATOR && regional % 2) join = true;
        else if (a == U_GCB_ZWJ && u_hasBinaryProperty(chars[i].value, UCHAR_EXTENDED_PICTOGRAPHIC)) {
            size_t j = i - 1;
            while (j > 0 && chars[j - 1].property == U_GCB_EXTEND) --j;
            join = j > 0 && u_hasBinaryProperty(chars[j - 1].value, UCHAR_EXTENDED_PICTOGRAPHIC);
        }
        if (!join) boundaries.push_back(chars[i].start);
        regional = b == U_GCB_REGIONAL_INDICATOR ? regional + 1 : 0;
    }
    boundaries.push_back(text.length());
    return boundaries;
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

UnigramVocabulary::UnigramVocabulary(const json &tokenizer, int bos, int eos, int unk,
                                     size_t context_length)
    : bos(bos), eos(eos), unk(unk), context_length(context_length) {
    check(tokenizer.contains("model") && tokenizer["model"].contains("vocab") &&
          tokenizer["model"].value("type", std::string("Unigram")) == "Unigram",
          "expected SentencePiece Unigram tokenizer");
    const auto &vocab = tokenizer["model"]["vocab"];
    check(vocab.is_array() && !vocab.empty() && vocab.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
          "expected Unigram tokenizer vocabulary");
    unknown_score = std::numeric_limits<double>::infinity();
    for (size_t id = 0; id < vocab.size(); ++id) {
        const auto &item = vocab[id];
        check(item.is_array() && item.size() == 2 && item[0].is_string(), "invalid Unigram vocabulary entry");
        const auto token = item[0].get<std::string>();
        const double score = item[1].get<double>();
        check(!token.empty() && std::isfinite(score), "invalid Unigram vocabulary entry");
        check(pieces.emplace(token, Piece{static_cast<int>(id), score}).second, "duplicate Unigram vocabulary entry");
        longest_piece = std::max(longest_piece, token.size());
        unknown_score = std::min(unknown_score, score);
    }
    unknown_score -= 10.0;
    check(bos >= 0 && eos >= 0 && unk >= 0 && static_cast<size_t>(std::max({bos, eos, unk})) < vocab.size() &&
          pieces.count("▁") != 0 && pieces.count("<unk>") != 0 && longest_piece > 0,
          "invalid Unigram tokenizer vocabulary");
    check(tokenizer["model"].at("unk_id").get<int>() == unk && !tokenizer["model"].value("byte_fallback", false),
          "unsupported Unigram unknown-token configuration");
    check(tokenizer.value("padding", json()).is_null() && tokenizer.value("truncation", json()).is_null(),
          "tokenizer-side padding/truncation is unsupported");
    const auto pre = tokenizer.value("pre_tokenizer", json());
    check(pre.is_object() && pre.value("type", std::string()) == "Sequence" &&
          pre.contains("pretokenizers") && pre["pretokenizers"].is_array() && pre["pretokenizers"].size() == 2,
          "expected WhitespaceSplit/Metaspace pre-tokenizer");
    const auto &steps = pre["pretokenizers"];
    check(steps[0].value("type", std::string()) == "WhitespaceSplit" &&
          steps[1].value("type", std::string()) == "Metaspace" &&
          steps[1].value("replacement", std::string()) == "▁" && steps[1].value("split", true) &&
          steps[1].value("add_prefix_space", true) && steps[1].value("prepend_scheme", std::string("always")) == "always",
          "unsupported Unigram pre-tokenizer");
    const std::string bos_piece = vocab.at(bos).at(0), eos_piece = vocab.at(eos).at(0);
    const auto post = tokenizer.value("post_processor", json());
    const json expected_single = json::array({
        {{"SpecialToken", {{"id", bos_piece}, {"type_id", 0}}}},
        {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
        {{"SpecialToken", {{"id", eos_piece}, {"type_id", 0}}}}});
    check(post.is_object() && post.value("type", std::string()) == "TemplateProcessing" &&
          post.value("single", json()) == expected_single && post.contains("special_tokens"),
          "unsupported Unigram post-processor");
    for (int id : {bos, eos}) {
        const std::string piece = vocab.at(id).at(0);
        check(post["special_tokens"].contains(piece) &&
              post["special_tokens"][piece].value("ids", json()) == json::array({id}),
              "invalid Unigram post-processor special token");
    }
    if (tokenizer.contains("added_tokens")) for (const auto &item : tokenizer["added_tokens"]) {
        check(item.value("special", false), "unsupported non-special added Unigram token");
        if (item.value("special", false)) {
            const auto content = item.at("content").get<std::string>();
            const int id = item.at("id").get<int>();
            check(!content.empty() && id >= 0 && static_cast<size_t>(id) < vocab.size() && vocab.at(id).at(0) == content,
                  "invalid Unigram special token");
            check(!item.value("normalized", false) && !item.value("lstrip", false) &&
                  !item.value("rstrip", false) && !item.value("single_word", false),
                  "unsupported Unigram special-token matching");
            special.emplace(content, id);
        }
    }
    const auto normalizer = tokenizer.value("normalizer", json());
    if (normalizer.is_null()) return;
    const auto type = normalizer.at("type").get<std::string>();
    check(type == "Precompiled", "unsupported Unigram tokenizer normalizer");
    // SentencePiece's serialized character map is a little-endian Darts trie
    // followed by NUL-terminated UTF-8 replacements. It is checkpoint-specific:
    // generic NFKC does not reproduce XLM-R's handling of format/control chars.
    const auto bytes = decode_charsmap(normalizer.at("precompiled_charsmap").get<std::string>());
    check(bytes.size() >= 8, "truncated tokenizer charsmap");
    const size_t trie_bytes = read_le32(bytes.data());
    check(trie_bytes >= 4 && trie_bytes % 4 == 0 && trie_bytes < bytes.size() - 4 && bytes.back() == 0,
          "invalid tokenizer charsmap layout");
    normalization_trie.reserve(trie_bytes / 4);
    for (size_t i = 4; i < 4 + trie_bytes; i += 4) normalization_trie.push_back(read_le32(bytes.data() + i));
    normalization_replacements.assign(reinterpret_cast<const char *>(bytes.data() + 4 + trie_bytes), bytes.size() - 4 - trie_bytes);
    check(trie_offset(normalization_trie[0]) < normalization_trie.size(), "invalid tokenizer charsmap root");
}

bool UnigramVocabulary::transform(const std::string &text, std::string &result) const {
    size_t node = trie_offset(normalization_trie[0]);
    for (unsigned char label : text) {
        if (!label) break;
        node ^= label;
        check(node < normalization_trie.size(), "invalid tokenizer charsmap transition");
        const uint32_t unit = normalization_trie[node];
        if ((unit & 0x800000ffu) != label) return false;
        node ^= trie_offset(unit);
        check(node < normalization_trie.size(), "invalid tokenizer charsmap transition");
        if (unit & (1u << 8)) {
            const size_t replacement = normalization_trie[node] & 0x7fffffffu;
            check(replacement < normalization_replacements.size(), "invalid tokenizer charsmap replacement");
            const auto end = normalization_replacements.find('\0', replacement);
            check(end != std::string::npos, "unterminated tokenizer charsmap replacement");
            result.append(normalization_replacements, replacement, end - replacement);
            return true;
        }
    }
    return false;
}

std::string UnigramVocabulary::normalize(const std::string &text) const {
    if (normalization_trie.empty()) return text;
    // HF's Precompiled normalizer intentionally tries each short (<6 UTF-8
    // bytes) extended grapheme first, then individual scalars, taking the FIRST
    // trie match. Longest-prefix SentencePiece normalization differs, notably
    // for decomposed Hangul and combining-character sequences.
    const auto unicode = icu::UnicodeString::fromUTF8(text);
    const auto boundaries = normalizer_boundaries(unicode);
    std::string result;
    for (size_t cluster = 1; cluster < boundaries.size(); ++cluster) {
        const auto grapheme = unicode.tempSubStringBetween(boundaries[cluster - 1], boundaries[cluster]);
        std::string bytes; grapheme.toUTF8String(bytes);
        if (bytes.size() < 6 && transform(bytes, result)) continue;
        for (int32_t i = 0; i < grapheme.length();) {
            const int32_t next = grapheme.moveIndex32(i, 1);
            std::string part; grapheme.tempSubStringBetween(i, next).toUTF8String(part);
            if (!transform(part, result)) result += part;
            i = next;
        }
    }
    return result;
}

void UnigramVocabulary::encode_word(const std::string &text, std::vector<int> &ids) const {
    if (text.empty()) return;
    std::vector<size_t> boundaries{0};
    for (size_t pos = 1; pos < text.size(); ++pos)
        if ((static_cast<unsigned char>(text[pos]) & 0xc0) != 0x80) boundaries.push_back(pos);
    boundaries.push_back(text.size());
    const size_t count = boundaries.size() - 1;
    struct State { double score; int previous; int id; };
    std::vector<State> best(count + 1, {-std::numeric_limits<double>::infinity(), -1, unk});
    best[0].score = 0;
    for (size_t begin = 0; begin < count; ++begin) {
        if (!std::isfinite(best[begin].score)) continue;
        bool single_character = false;
        for (size_t end = begin + 1; end <= count && boundaries[end] - boundaries[begin] <= longest_piece; ++end) {
            const auto piece = pieces.find(text.substr(boundaries[begin], boundaries[end] - boundaries[begin]));
            if (piece == pieces.end()) continue;
            single_character = single_character || end == begin + 1;
            const double score = best[begin].score + piece->second.score;
            if (score > best[end].score) best[end] = {score, static_cast<int>(begin), piece->second.id};
        }
        const double score = best[begin].score + unknown_score;
        if (!single_character && score > best[begin + 1].score)
            best[begin + 1] = {score, static_cast<int>(begin), unk};
    }
    check(best[count].previous >= 0, "Unigram tokenizer could not encode text");
    std::vector<int> reversed;
    for (int pos = static_cast<int>(count); pos > 0; pos = best[pos].previous) reversed.push_back(best[pos].id);
    bool previous_unknown = false;
    for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
        if (*it != unk || !previous_unknown) ids.push_back(*it);
        previous_unknown = *it == unk;
    }
}

std::vector<int> UnigramVocabulary::tokenize(const std::string &text, size_t max_length) const {
    check(max_length >= 2 && max_length <= context_length, "invalid maximum text length");
    std::vector<int> ids{bos};
    size_t offset = 0;
    while (offset < text.size()) {
        size_t special_at = std::string::npos, special_end = std::string::npos;
        int special_id = -1;
        for (const auto &entry : special) {
            const auto pos = text.find(entry.first, offset);
            if (pos < special_at) { special_at = pos; special_end = pos + entry.first.size(); special_id = entry.second; }
        }
        const size_t stop = special_at == std::string::npos ? text.size() : special_at;
        if (stop > offset) {
            const auto normalized = icu::UnicodeString::fromUTF8(normalize(text.substr(offset, stop - offset)));
            int32_t word_start = 0;
            bool in_word = false;
            for (int32_t i = 0; i < normalized.length();) {
                const int32_t next = normalized.moveIndex32(i, 1);
                if (u_isUWhiteSpace(normalized.char32At(i))) {
                    if (in_word) {
                        std::string word; normalized.tempSubStringBetween(word_start, i).toUTF8String(word);
                        encode_word("▁" + word, ids); in_word = false;
                    }
                } else if (!in_word) { word_start = i; in_word = true; }
                i = next;
            }
            if (in_word) { std::string word; normalized.tempSubStringBetween(word_start).toUTF8String(word); encode_word("▁" + word, ids); }
        }
        if (special_at == std::string::npos) break;
        ids.push_back(special_id); offset = special_end;
    }
    if (ids.size() > max_length - 1) ids.resize(max_length - 1);
    ids.push_back(eos);
    return ids;
}
} // namespace clap::detail
