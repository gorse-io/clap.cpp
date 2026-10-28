#include "vocab.h"
#include <stdexcept>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>

namespace clap::detail {
using json = nlohmann::json;
namespace {
std::string utf8(int cp) {
    icu::UnicodeString u; u.append(static_cast<UChar32>(cp));
    std::string s; u.toUTF8String(s); return s;
}

std::vector<unsigned char> decode_charsmap(const std::string &encoded) {
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (!(!encoded.empty() && encoded.size() % 4 == 0))
        throw std::runtime_error("CLAP: invalid tokenizer charsmap encoding");
    std::vector<unsigned char> bytes;
    for (size_t i = 0; i < encoded.size(); i += 4) {
        uint32_t value = 0;
        int padding = 0;
        for (size_t j = 0; j < 4; ++j) {
            const char c = encoded[i + j];
            if (c == '=') {
                if (!(i + 4 == encoded.size() && j >= 2))
                    throw std::runtime_error("CLAP: invalid tokenizer charsmap padding");
                ++padding; value <<= 6;
            } else {
                const auto digit = alphabet.find(c);
                if (!(!padding && digit != std::string::npos))
                    throw std::runtime_error("CLAP: invalid tokenizer charsmap encoding");
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
    if (tokenizer.at("model").at("type") != "BPE")
        throw std::runtime_error("CLAP: expected byte-level BPE");
    for (auto it = tokenizer["model"]["vocab"].begin(); it != tokenizer["model"]["vocab"].end(); ++it)
        vocab.emplace(it.key(), it.value().get<int>());
    int rank = 0;
    for (const auto &entry : tokenizer["model"]["merges"]) {
        std::string x, y;
        if (entry.is_array()) { x = entry.at(0); y = entry.at(1); }
        else {
            const std::string s = entry; const auto space = s.find(' ');
            if (space == std::string::npos)
                throw std::runtime_error("CLAP: invalid BPE merge");
            x = s.substr(0, space); y = s.substr(space + 1);
        }
        ranks.emplace(std::make_pair(x, y), rank++);
    }
    if (tokenizer.contains("added_tokens")) for (const auto &v : tokenizer["added_tokens"]) {
        if (v.at("content").get<std::string>().empty())
            throw std::runtime_error("CLAP: empty added token");
        special.push_back(v);
    }
    int extra = 256;
    for (int i = 0; i < 256; ++i) {
        const bool regular = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174;
        byte_chars[i] = utf8(regular ? i : extra++);
    }
    UErrorCode status = U_ZERO_ERROR;
    pattern.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"), 0, status));
    if (U_FAILURE(status))
        throw std::runtime_error("CLAP: cannot compile tokenizer Unicode regex");
}

void ByteBpeVocabulary::encode_piece(const std::string &text, std::vector<int> &ids) const {
    UErrorCode status = U_ZERO_ERROR;
    const auto unicode = icu::UnicodeString::fromUTF8(text);
    std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(unicode, status));
    if (U_FAILURE(status))
        throw std::runtime_error("CLAP: tokenizer regex failed");
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
            if (it == vocab.end())
                throw std::runtime_error("CLAP: tokenizer has no token for byte sequence");
            ids.push_back(it->second);
        }
    }
    if (U_FAILURE(status))
        throw std::runtime_error("CLAP: tokenizer regex failed");
}

std::vector<int> ByteBpeVocabulary::tokenize(const std::string &text, size_t max_length) const {
    if (!(max_length >= 2 && max_length <= context_length))
        throw std::runtime_error("CLAP: invalid maximum text length");
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

GemmaVocabulary::GemmaVocabulary(const json &metadata, size_t context_length)
    : bos(-1), eos(-1), unk(-1),
      add_bos(metadata.value("tokenizer.ggml.add_bos_token", false)),
      add_eos(metadata.value("tokenizer.ggml.add_eos_token", false)), context_length(context_length) {
    if (metadata.at("tokenizer.ggml.model") != "gemma4" || !context_length ||
        metadata.value("tokenizer.ggml.add_space_prefix", false) ||
        metadata.value("tokenizer.ggml.remove_extra_whitespaces", false))
        throw std::runtime_error("CLAP: unsupported Gemma tokenizer configuration");
    const auto &tokens = metadata.at("tokenizer.ggml.tokens");
    const auto &types = metadata.at("tokenizer.ggml.token_type");
    const auto &merges = metadata.at("tokenizer.ggml.merges");
    if (!tokens.is_array() || tokens.empty() || tokens.size() > size_t(std::numeric_limits<int>::max()) ||
        !types.is_array() || types.size() != tokens.size() || !merges.is_array())
        throw std::runtime_error("CLAP: invalid Gemma tokenizer arrays");
    auto bounded_integer = [](const json &value, uint64_t minimum, uint64_t maximum, const char *error) {
        // Check the original signed/unsigned JSON value before narrowing to int.
        if (!value.is_number_integer() ||
            (!value.is_number_unsigned() && value.get<int64_t>() < 0) ||
            value.get<uint64_t>() < minimum || value.get<uint64_t>() > maximum)
            throw std::runtime_error(error);
        return value.get<int>();
    };
    auto special_id = [&](const char *key) {
        // -1 is an internal sentinel for absent metadata, never an explicit id.
        return metadata.contains(key)
            ? bounded_integer(metadata.at(key), 0, tokens.size() - 1, "CLAP: invalid Gemma special token id")
            : -1;
    };
    bos = special_id("tokenizer.ggml.bos_token_id");
    eos = special_id("tokenizer.ggml.eos_token_id");
    unk = special_id("tokenizer.ggml.unknown_token_id");
    if (metadata.contains("tokenizer.ggml.scores")) {
        const auto &scores = metadata.at("tokenizer.ggml.scores");
        if (!scores.is_array() || scores.size() != tokens.size())
            throw std::runtime_error("CLAP: invalid Gemma tokenizer scores");
        for (const auto &s : scores) if (!s.is_number() || !std::isfinite(s.get<double>()))
            throw std::runtime_error("CLAP: invalid Gemma tokenizer score");
    }
    std::fill(std::begin(bytes), std::end(bytes), -1);
    for (size_t id = 0; id < tokens.size(); ++id) {
        const auto token = tokens.at(id).get<std::string>();
        const int type = bounded_integer(types.at(id), 1, 6, "CLAP: invalid Gemma token type");
        if (token.empty())
            throw std::runtime_error("CLAP: invalid Gemma token");
        if (!pieces.emplace(token, static_cast<int>(id)).second)
            throw std::runtime_error("CLAP: duplicate Gemma token");
        if (type == 2 || type == 3 || type == 4) special.emplace_back(token, static_cast<int>(id));
        if (type == 6) {
            if (token.size() != 6 || token.compare(0, 3, "<0x") || token[5] != '>')
                throw std::runtime_error("CLAP: invalid Gemma byte token");
            const auto digits = std::string("0123456789ABCDEF");
            const auto a = digits.find(token[3]), b = digits.find(token[4]);
            if (a == std::string::npos || b == std::string::npos)
                throw std::runtime_error("CLAP: invalid Gemma byte token");
            bytes[a * 16 + b] = static_cast<int>(id);
        }
    }
    for (int id : bytes) if (id < 0) throw std::runtime_error("CLAP: incomplete Gemma byte fallback");
    auto valid = [&](int id) { return id >= 0 && size_t(id) < tokens.size(); };
    if ((add_bos && !valid(bos)) || (add_eos && !valid(eos)) || (unk != -1 && !valid(unk)))
        throw std::runtime_error("CLAP: invalid Gemma special token id");
    if (merges.size() > size_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("CLAP: too many Gemma merges");
    int rank = 0;
    for (const auto &item : merges) {
        const auto merge = item.get<std::string>();
        const auto split = merge.find(' ');
        if (split == std::string::npos || split == 0 || split + 1 == merge.size())
            throw std::runtime_error("CLAP: invalid Gemma merge");
        auto pair = std::make_pair(merge.substr(0, split), merge.substr(split + 1));
        if (!pieces.count(pair.first) || !pieces.count(pair.second) || !pieces.count(pair.first + pair.second))
            throw std::runtime_error("CLAP: Gemma merge references missing token");
        ranks.emplace(std::move(pair), rank++);
    }
    std::sort(special.begin(), special.end(), [](const auto &a, const auto &b) {
        return a.first.size() > b.first.size();
    });
}

void GemmaVocabulary::encode_piece(const std::string &text, std::vector<int> &ids) const {
    // Only ASCII space is normalized. Preserve all other Unicode, whitespace,
    // repeated spaces, controls and combining marks exactly (no ICU NFKC).
    std::string normalized;
    for (char c : text) { if (c == ' ') normalized += "▁"; else normalized += c; }
    if (normalized.empty()) return;
    struct Symbol { size_t start, size; int previous, next; };
    struct Merge { int rank, left, right; size_t left_size, right_size; };
    struct Compare { bool operator()(const Merge &a, const Merge &b) const {
        return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
    }};
    std::vector<Symbol> symbols;
    for (size_t pos = 0; pos < normalized.size();) {
        const auto c = static_cast<unsigned char>(normalized[pos]);
        size_t length = c < 0x80 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 1;
        length = std::min(length, normalized.size() - pos);
        for (size_t k = 1; k < length; ++k) if ((static_cast<unsigned char>(normalized[pos+k]) & 0xc0) != 0x80) { length = 1; break; }
        const int i = static_cast<int>(symbols.size());
        symbols.push_back({pos, length, i-1, i+1}); pos += length;
    }
    symbols.back().next = -1;
    std::priority_queue<Merge, std::vector<Merge>, Compare> queue;
    auto add = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        const auto &a = symbols[left], &b = symbols[right];
        auto it = ranks.find({normalized.substr(a.start,a.size), normalized.substr(b.start,b.size)});
        if (it != ranks.end()) queue.push({it->second,left,right,a.size,b.size});
    };
    for (size_t i = 1; i < symbols.size(); ++i) add(static_cast<int>(i)-1,static_cast<int>(i));
    while (!queue.empty()) {
        const auto merge = queue.top(); queue.pop();
        auto &a = symbols[merge.left], &b = symbols[merge.right];
        if (a.size != merge.left_size || b.size != merge.right_size || a.next != merge.right) continue;
        a.size += b.size; b.size = 0; a.next = b.next;
        if (a.next >= 0) symbols[a.next].previous = merge.left;
        add(a.previous, merge.left); add(merge.left,a.next);
    }
    for (int i = 0; i >= 0; i = symbols[i].next) {
        const auto &s = symbols[i];
        const auto it = pieces.find(normalized.substr(s.start,s.size));
        if (it != pieces.end()) ids.push_back(it->second);
        else for (size_t j = s.start; j < s.start+s.size; ++j) ids.push_back(bytes[static_cast<unsigned char>(normalized[j])]);
    }
}

std::vector<int> GemmaVocabulary::tokenize(const std::string &text, size_t max_length) const {
    const size_t reserved = size_t(add_bos) + size_t(add_eos);
    if (!max_length || max_length < reserved || max_length > context_length)
        throw std::runtime_error("CLAP: invalid maximum text length");
    std::vector<int> ids;
    if (add_bos) ids.push_back(bos);
    size_t pos = 0;
    while (pos < text.size()) {
        size_t next = std::string::npos;
        const std::pair<std::string,int> *match = nullptr;
        for (const auto &s : special) {
            const auto found = text.find(s.first,pos);
            if (found < next) { next = found; match = &s; }
        }
        if (!match) { encode_piece(text.substr(pos), ids); break; }
        encode_piece(text.substr(pos,next-pos),ids);
        ids.push_back(match->second); pos = next + match->first.size();
    }
    if (ids.size() > max_length - size_t(add_eos)) ids.resize(max_length-size_t(add_eos));
    if (add_eos) ids.push_back(eos);
    return ids;
}

UnigramVocabulary::UnigramVocabulary(const json &tokenizer, int bos, int eos, int unk,
                                     size_t context_length)
    : bos(bos), eos(eos), unk(unk), context_length(context_length) {
    if (!(tokenizer.contains("model") && tokenizer["model"].contains("vocab") &&
          tokenizer["model"].value("type", std::string("Unigram")) == "Unigram"))
        throw std::runtime_error("CLAP: expected SentencePiece Unigram tokenizer");
    const auto &vocab = tokenizer["model"]["vocab"];
    if (!(vocab.is_array() && !vocab.empty() && vocab.size() <= static_cast<size_t>(std::numeric_limits<int>::max())))
        throw std::runtime_error("CLAP: expected Unigram tokenizer vocabulary");
    unknown_score = std::numeric_limits<double>::infinity();
    for (size_t id = 0; id < vocab.size(); ++id) {
        const auto &item = vocab[id];
        if (!(item.is_array() && item.size() == 2 && item[0].is_string()))
            throw std::runtime_error("CLAP: invalid Unigram vocabulary entry");
        const auto token = item[0].get<std::string>();
        const double score = item[1].get<double>();
        if (!(!token.empty() && std::isfinite(score)))
            throw std::runtime_error("CLAP: invalid Unigram vocabulary entry");
        if (!(pieces.emplace(token, Piece{static_cast<int>(id), score}).second))
            throw std::runtime_error("CLAP: duplicate Unigram vocabulary entry");
        longest_piece = std::max(longest_piece, token.size());
        unknown_score = std::min(unknown_score, score);
    }
    unknown_score -= 10.0;
    if (!(bos >= 0 && eos >= 0 && unk >= 0 && static_cast<size_t>(std::max({bos, eos, unk})) < vocab.size() &&
          pieces.count("▁") != 0 && pieces.count("<unk>") != 0 && longest_piece > 0))
        throw std::runtime_error("CLAP: invalid Unigram tokenizer vocabulary");
    if (!(tokenizer["model"].at("unk_id").get<int>() == unk && !tokenizer["model"].value("byte_fallback", false)))
        throw std::runtime_error("CLAP: unsupported Unigram unknown-token configuration");
    if (!(tokenizer.value("padding", json()).is_null() && tokenizer.value("truncation", json()).is_null()))
        throw std::runtime_error("CLAP: tokenizer-side padding/truncation is unsupported");
    const auto pre = tokenizer.value("pre_tokenizer", json());
    if (!(pre.is_object() && pre.value("type", std::string()) == "Sequence" &&
          pre.contains("pretokenizers") && pre["pretokenizers"].is_array() && pre["pretokenizers"].size() == 2))
        throw std::runtime_error("CLAP: expected WhitespaceSplit/Metaspace pre-tokenizer");
    const auto &steps = pre["pretokenizers"];
    if (!(steps[0].value("type", std::string()) == "WhitespaceSplit" &&
          steps[1].value("type", std::string()) == "Metaspace" &&
          steps[1].value("replacement", std::string()) == "▁" && steps[1].value("split", true) &&
          steps[1].value("add_prefix_space", true) && steps[1].value("prepend_scheme", std::string("always")) == "always"))
        throw std::runtime_error("CLAP: unsupported Unigram pre-tokenizer");
    const std::string bos_piece = vocab.at(bos).at(0), eos_piece = vocab.at(eos).at(0);
    const auto post = tokenizer.value("post_processor", json());
    const json expected_single = json::array({
        {{"SpecialToken", {{"id", bos_piece}, {"type_id", 0}}}},
        {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
        {{"SpecialToken", {{"id", eos_piece}, {"type_id", 0}}}}});
    if (!(post.is_object() && post.value("type", std::string()) == "TemplateProcessing" &&
          post.value("single", json()) == expected_single && post.contains("special_tokens")))
        throw std::runtime_error("CLAP: unsupported Unigram post-processor");
    for (int id : {bos, eos}) {
        const std::string piece = vocab.at(id).at(0);
        if (!(post["special_tokens"].contains(piece) &&
              post["special_tokens"][piece].value("ids", json()) == json::array({id})))
            throw std::runtime_error("CLAP: invalid Unigram post-processor special token");
    }
    if (tokenizer.contains("added_tokens")) for (const auto &item : tokenizer["added_tokens"]) {
        if (!(item.value("special", false)))
            throw std::runtime_error("CLAP: unsupported non-special added Unigram token");
        if (item.value("special", false)) {
            const auto content = item.at("content").get<std::string>();
            const int id = item.at("id").get<int>();
            if (!(!content.empty() && id >= 0 && static_cast<size_t>(id) < vocab.size() && vocab.at(id).at(0) == content))
                throw std::runtime_error("CLAP: invalid Unigram special token");
            if (!(!item.value("normalized", false) && !item.value("lstrip", false) &&
                  !item.value("rstrip", false) && !item.value("single_word", false)))
                throw std::runtime_error("CLAP: unsupported Unigram special-token matching");
            special.emplace(content, id);
        }
    }
    const auto normalizer = tokenizer.value("normalizer", json());
    if (normalizer.is_null()) return;
    const auto type = normalizer.at("type").get<std::string>();
    if (type != "Precompiled")
        throw std::runtime_error("CLAP: unsupported Unigram tokenizer normalizer");
    // SentencePiece's serialized character map is a little-endian Darts trie
    // followed by NUL-terminated UTF-8 replacements. It is checkpoint-specific:
    // generic NFKC does not reproduce XLM-R's handling of format/control chars.
    const auto bytes = decode_charsmap(normalizer.at("precompiled_charsmap").get<std::string>());
    if (!(bytes.size() >= 8))
        throw std::runtime_error("CLAP: truncated tokenizer charsmap");
    const size_t trie_bytes = read_le32(bytes.data());
    if (!(trie_bytes >= 4 && trie_bytes % 4 == 0 && trie_bytes < bytes.size() - 4 && bytes.back() == 0))
        throw std::runtime_error("CLAP: invalid tokenizer charsmap layout");
    normalization_trie.reserve(trie_bytes / 4);
    for (size_t i = 4; i < 4 + trie_bytes; i += 4) normalization_trie.push_back(read_le32(bytes.data() + i));
    normalization_replacements.assign(reinterpret_cast<const char *>(bytes.data() + 4 + trie_bytes), bytes.size() - 4 - trie_bytes);
    if (!(trie_offset(normalization_trie[0]) < normalization_trie.size()))
        throw std::runtime_error("CLAP: invalid tokenizer charsmap root");
}

bool UnigramVocabulary::transform(const std::string &text, std::string &result) const {
    size_t node = trie_offset(normalization_trie[0]);
    for (unsigned char label : text) {
        if (!label) break;
        node ^= label;
        if (!(node < normalization_trie.size()))
            throw std::runtime_error("CLAP: invalid tokenizer charsmap transition");
        const uint32_t unit = normalization_trie[node];
        if ((unit & 0x800000ffu) != label) return false;
        node ^= trie_offset(unit);
        if (!(node < normalization_trie.size()))
            throw std::runtime_error("CLAP: invalid tokenizer charsmap transition");
        if (unit & (1u << 8)) {
            const size_t replacement = normalization_trie[node] & 0x7fffffffu;
            if (!(replacement < normalization_replacements.size()))
                throw std::runtime_error("CLAP: invalid tokenizer charsmap replacement");
            const auto end = normalization_replacements.find('\0', replacement);
            if (end == std::string::npos)
                throw std::runtime_error("CLAP: unterminated tokenizer charsmap replacement");
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
    if (!(best[count].previous >= 0))
        throw std::runtime_error("CLAP: Unigram tokenizer could not encode text");
    std::vector<int> reversed;
    for (int pos = static_cast<int>(count); pos > 0; pos = best[pos].previous) reversed.push_back(best[pos].id);
    bool previous_unknown = false;
    for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
        if (*it != unk || !previous_unknown) ids.push_back(*it);
        previous_unknown = *it == unk;
    }
}

std::vector<int> UnigramVocabulary::tokenize(const std::string &text, size_t max_length) const {
    if (!(max_length >= 2 && max_length <= context_length))
        throw std::runtime_error("CLAP: invalid maximum text length");
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
